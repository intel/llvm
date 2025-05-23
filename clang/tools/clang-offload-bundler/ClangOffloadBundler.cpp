//===-- clang-offload-bundler/ClangOffloadBundler.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a stand-alone clang-offload-bundler tool using the
/// OffloadBundler API.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Cuda.h"
#include "clang/Basic/TargetID.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/OffloadBundler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace llvm::object;
using namespace clang;

static void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-offload-bundler") << '\n';
}

int main(int argc, const char **argv) {

  cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

  // Mark all our options with this category, everything else (except for
  // -version and -help) will be hidden.
  cl::OptionCategory
    ClangOffloadBundlerCategory("clang-offload-bundler options");
  cl::list<std::string>
    InputFileNames("input",
                   cl::desc("Input file."
                            " Can be specified multiple times "
                            "for multiple input files."),
                   cl::cat(ClangOffloadBundlerCategory));
  cl::list<std::string>
    InputFileNamesDeprecatedOpt("inputs", cl::CommaSeparated,
                                cl::desc("[<input file>,...] (deprecated)"),
                                cl::cat(ClangOffloadBundlerCategory));
  cl::list<std::string>
    OutputFileNames("output",
                    cl::desc("Output file."
                             " Can be specified multiple times "
                             "for multiple output files."),
                    cl::cat(ClangOffloadBundlerCategory));
  cl::list<std::string>
    OutputFileNamesDeprecatedOpt("outputs", cl::CommaSeparated,
                                 cl::desc("[<output file>,...] (deprecated)"),
                                 cl::cat(ClangOffloadBundlerCategory));
  cl::list<std::string>
    TargetNames("targets", cl::CommaSeparated,
                cl::desc("[<offload kind>-<target triple>,...]"),
                cl::cat(ClangOffloadBundlerCategory));

  cl::list<std::string> ExcludedTargetNames(
      "excluded-targets", cl::CommaSeparated,
      cl::desc("[<target name>,...]. List of targets that are excluded from "
               "unbundling."),
      cl::cat(ClangOffloadBundlerCategory));

  cl::opt<std::string> FilesType(
      "type", cl::Required,
      cl::desc(
          "Type of the files to be bundled/unbundled/checked.\n"
          "Current supported types are:\n"
          "  i   - cpp-output\n"
          "  ii  - c++-cpp-output\n"
          "  cui - cuda/hip-output\n"
          "  hipi - hip-cpp-output\n"
          "  d   - dependency\n"
          "  ll  - llvm\n"
          "  bc  - llvm-bc\n"
          "  s   - assembler\n"
          "  o   - object\n"
          "  gch - precompiled-header\n"
          "  ast - clang AST file\n"
          "  a   - archive of objects\n"
          "  ao  - archive with one object; output is an unbundled object\n"
          "  aoo - archive; output file is a list of unbundled objects\n"),
      cl::cat(ClangOffloadBundlerCategory));

  cl::opt<bool>
    Unbundle("unbundle",
             cl::desc("Unbundle bundled file into several output files.\n"),
             cl::init(false), cl::cat(ClangOffloadBundlerCategory));

  cl::opt<bool> CheckSection("check-section",
                                  cl::desc("Check if the section exists.\n"),
                                  cl::init(false),
                                  cl::cat(ClangOffloadBundlerCategory));

  cl::opt<bool>
    ListBundleIDs("list", cl::desc("List bundle IDs in the bundled file.\n"),
                  cl::init(false), cl::cat(ClangOffloadBundlerCategory));

  cl::opt<bool> PrintExternalCommands(
    "###",
    cl::desc("Print any external commands that are to be executed "
             "instead of actually executing them - for testing purposes.\n"),
    cl::init(false), cl::cat(ClangOffloadBundlerCategory));

  cl::opt<bool>
    AllowMissingBundles("allow-missing-bundles",
                        cl::desc("Create empty files if bundles are missing "
                                 "when unbundling.\n"),
                        cl::init(false), cl::cat(ClangOffloadBundlerCategory));

  cl::opt<unsigned>
    BundleAlignment("bundle-align",
                    cl::desc("Alignment of bundle for binary files"),
                    cl::init(1), cl::cat(ClangOffloadBundlerCategory));

  cl::opt<bool>
    AddTargetSymbols("add-target-symbols-to-bundled-object",
                     cl::desc("Add .tgtsym section with target symbol names to "
                              "the output file when bundling object files.\n"),
                     cl::init(true), cl::cat(ClangOffloadBundlerCategory));

  cl::opt<bool> CheckInputArchive(
      "check-input-archive",
      cl::desc("Check if input heterogeneous archive is "
               "valid in terms of TargetID rules.\n"),
      cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool> HipOpenmpCompatible(
    "hip-openmp-compatible",
    cl::desc("Treat hip and hipv4 offload kinds as "
             "compatible with openmp kind, and vice versa.\n"),
    cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool> Compress("compress",
                         cl::desc("Compress output file when bundling.\n"),
                         cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool> Verbose("verbose", cl::desc("Print debug information.\n"),
                        cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<int> CompressionLevel(
      "compression-level", cl::desc("Specify the compression level (integer)"),
      cl::value_desc("n"), cl::Optional, cl::cat(ClangOffloadBundlerCategory));

  // Process commandline options and report errors
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangOffloadBundlerCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to bundle several input files of the specified type <type> \n"
      "referring to the same source file but different targets into a single \n"
      "one. The resulting file can also be unbundled into different files by \n"
      "this tool if -unbundle is provided.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  // These calls are needed so that we can read bitcode correctly.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();

  /// Class to store bundler options in standard (non-cl::opt) data structures
  // Avoid using cl::opt variables after these assignments when possible
  OffloadBundlerConfig BundlerConfig;
  BundlerConfig.AllowMissingBundles = AllowMissingBundles;
  BundlerConfig.CheckInputArchive = CheckInputArchive;
  BundlerConfig.PrintExternalCommands = PrintExternalCommands;
  BundlerConfig.AddTargetSymbols = AddTargetSymbols;
  BundlerConfig.HipOpenmpCompatible = HipOpenmpCompatible;
  BundlerConfig.BundleAlignment = BundleAlignment;
  BundlerConfig.FilesType = FilesType;
  BundlerConfig.ObjcopyPath = "";
  // Do not override the default value Compress and Verbose in BundlerConfig.
  if (Compress.getNumOccurrences() > 0)
    BundlerConfig.Compress = Compress;
  if (Verbose.getNumOccurrences() > 0)
    BundlerConfig.Verbose = Verbose;
  if (CompressionLevel.getNumOccurrences() > 0)
    BundlerConfig.CompressionLevel = CompressionLevel;

  BundlerConfig.TargetNames = TargetNames;
  BundlerConfig.ExcludedTargetNames = ExcludedTargetNames;
  BundlerConfig.InputFileNames = InputFileNames;
  BundlerConfig.OutputFileNames = OutputFileNames;

  /// The index of the host input in the list of inputs.
  BundlerConfig.HostInputIndex = ~0u;

  /// Whether not having host target is allowed.
  BundlerConfig.AllowNoHost = false;

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    return 1;
  };

  auto doWork = [&](std::function<llvm::Error()> Work) {
    if (llvm::Error Err = Work()) {
      return reportError(std::move(Err));
    }
    return 0;
  };

  auto warningOS = [argv]() -> raw_ostream & {
    return WithColor::warning(errs(), StringRef(argv[0]));
  };

  /// Path to the current binary.
  std::string BundlerExecutable = argv[0];

  if (!llvm::sys::fs::exists(BundlerExecutable))
    BundlerExecutable =
      sys::fs::getMainExecutable(argv[0], &BundlerExecutable);

  // Find llvm-objcopy in order to create the bundle binary.
  ErrorOr<std::string> Objcopy = sys::findProgramByName(
    "llvm-objcopy",
    sys::path::parent_path(BundlerExecutable));
  if (!Objcopy)
    Objcopy = sys::findProgramByName("llvm-objcopy");
  if (!Objcopy)
    return reportError(createStringError(
        Objcopy.getError(), "unable to find 'llvm-objcopy' in path"));
  else
    BundlerConfig.ObjcopyPath = *Objcopy;

  if (InputFileNames.getNumOccurrences() != 0 &&
      InputFileNamesDeprecatedOpt.getNumOccurrences() != 0) {
    return reportError(createStringError(
        errc::invalid_argument,
        "-inputs and -input cannot be used together, use only -input instead"));
  }

  if (InputFileNamesDeprecatedOpt.size()) {
    warningOS() << "-inputs is deprecated, use -input instead\n";
    // temporary hack to support -inputs
    std::vector<std::string> &s = InputFileNames;
    s.insert(s.end(), InputFileNamesDeprecatedOpt.begin(),
             InputFileNamesDeprecatedOpt.end());
  }
  BundlerConfig.InputFileNames = InputFileNames;

  if (OutputFileNames.getNumOccurrences() != 0 &&
      OutputFileNamesDeprecatedOpt.getNumOccurrences() != 0) {
    return reportError(createStringError(errc::invalid_argument,
                                         "-outputs and -output cannot be used "
                                         "together, use only -output instead"));
  }

  if (OutputFileNamesDeprecatedOpt.size()) {
    warningOS() << "-outputs is deprecated, use -output instead\n";
    // temporary hack to support -outputs
    std::vector<std::string> &s = OutputFileNames;
    s.insert(s.end(), OutputFileNamesDeprecatedOpt.begin(),
             OutputFileNamesDeprecatedOpt.end());
  }
  BundlerConfig.OutputFileNames = OutputFileNames;

  if (ListBundleIDs) {
    if (Unbundle) {
      return reportError(
          createStringError(errc::invalid_argument,
                            "-unbundle and -list cannot be used together"));
    }
    if (InputFileNames.size() != 1) {
      return reportError(createStringError(
          errc::invalid_argument, "only one input file supported for -list"));
    }
    if (OutputFileNames.size()) {
      return reportError(createStringError(
          errc::invalid_argument, "-outputs option is invalid for -list"));
    }
    if (TargetNames.size()) {
      return reportError(createStringError(
          errc::invalid_argument, "-targets option is invalid for -list"));
    }

    return doWork([&]() {
      return OffloadBundler::ListBundleIDsInFile(InputFileNames.front(),
                                                 BundlerConfig);
    });
  }

  if (BundlerConfig.CheckInputArchive) {
    if (!Unbundle) {
      return reportError(createStringError(
          errc::invalid_argument, "-check-input-archive cannot be used while "
                                  "bundling"));
    }
    if (Unbundle && BundlerConfig.FilesType != "a") {
      return reportError(createStringError(
          errc::invalid_argument, "-check-input-archive can only be used for "
                                  "unbundling archives (-type=a)"));
    }
  }

  if (OutputFileNames.size() == 0 && !CheckSection) {
    return reportError(
        createStringError(errc::invalid_argument, "no output file specified!"));
  }

  if (TargetNames.getNumOccurrences() == 0) {
    return reportError(createStringError(
        errc::invalid_argument,
        "for the --targets option: must be specified at least once!"));
  }

  if (Unbundle && CheckSection) {
    reportError(createStringError(
        errc::invalid_argument,
        "-unbundle and -check-section are not compatible options"));
    return 1;
  }

  // -check-section
  if (CheckSection) {
    if (InputFileNames.size() != 1) {
      reportError(
          createStringError(errc::invalid_argument,
                            "only one input file supported in checking mode"));
    }
    if (TargetNames.size() != 1) {
      reportError(
          createStringError(errc::invalid_argument,
                            "only one target supported in checking mode"));
    }
    if (OutputFileNames.size() != 0) {
      reportError(createStringError(
          errc::invalid_argument, "no output file supported in checking mode"));
    }
  }
  // -unbundle
  else if (Unbundle) {
    if (InputFileNames.size() != 1) {
      return reportError(createStringError(
          errc::invalid_argument,
          "only one input file supported in unbundling mode"));
    }
    if (OutputFileNames.size() != TargetNames.size()) {
      return reportError(createStringError(
          errc::invalid_argument, "number of output files and targets should "
                                  "match in unbundling mode"));
    }
  }
  // no explicit option: bundle
  else {
    if (BundlerConfig.FilesType == "a") {
      return reportError(createStringError(errc::invalid_argument,
                                           "Archive files are only supported "
                                           "for unbundling"));
    }
    if (OutputFileNames.size() != 1) {
      return reportError(
          createStringError(errc::invalid_argument,
                            "only one output file supported in bundling mode"));
    }
    if (InputFileNames.size() != TargetNames.size()) {
      return reportError(createStringError(
          errc::invalid_argument,
          "number of input files and targets should match in bundling mode"));
    }
  }

  // check -excluded-targets without unbundle
  if (!ExcludedTargetNames.empty() && !Unbundle)
    reportError(createStringError(errc::invalid_argument,
                                  "-excluded-targets option should be used "
                                  "only in conjunction with -unbundle"));

  // Verify that the offload kinds and triples are known. We also check that we
  // have exactly one host target.
  unsigned Index = 0u;
  unsigned HostTargetNum = 0u;
  bool HIPOnly = true;
  llvm::DenseSet<StringRef> ParsedTargets;
  // Map {offload-kind}-{triple} to target IDs.
  std::map<std::string, std::set<StringRef>> TargetIDs;
  for (StringRef Target : TargetNames) {
    if (!ParsedTargets.insert(Target).second) {
      return reportError(createStringError(
          errc::invalid_argument, "Duplicate targets are not allowed"));
    }

    if (!checkOffloadBundleID(Target)) {
      return reportError(createStringError(
          errc::invalid_argument,
          "Targets need to follow the format '<offload kind>-<target triple>', "
          "where '<target triple>' follows the format "
          "'<kind>-<arch>-<vendor>-<os>-<env>[-<target id>[:target "
          "features]]'."));
    }

    auto OffloadInfo = OffloadTargetInfo(Target, BundlerConfig);
    bool KindIsValid = OffloadInfo.isOffloadKindValid();
    bool TripleIsValid = OffloadInfo.isTripleValid();

    if (!KindIsValid || !TripleIsValid) {
      SmallVector<char, 128u> Buf;
      raw_svector_ostream Msg(Buf);
      Msg << "invalid target '" << Target << "'";
      if (!KindIsValid)
        Msg << ", unknown offloading kind '" << OffloadInfo.OffloadKind << "'";
      if (!TripleIsValid)
        Msg << ", unknown target triple '" << OffloadInfo.Triple.str() << "'";
      return reportError(createStringError(errc::invalid_argument, Msg.str()));
    }

    TargetIDs[OffloadInfo.OffloadKind.str() + "-" + OffloadInfo.Triple.str()]
        .insert(OffloadInfo.TargetID);
    if (KindIsValid && OffloadInfo.hasHostKind()) {
      ++HostTargetNum;
      // Save the index of the input that refers to the host.
      BundlerConfig.HostInputIndex = Index;
    }

    if (OffloadInfo.OffloadKind != "hip" && OffloadInfo.OffloadKind != "hipv4")
      HIPOnly = false;

    ++Index;
  }
  for (const auto &TargetID : TargetIDs) {
    if (auto ConflictingTID =
            clang::getConflictTargetIDCombination(TargetID.second)) {
      SmallVector<char, 128u> Buf;
      raw_svector_ostream Msg(Buf);
      Msg << "Cannot bundle inputs with conflicting targets: '"
          << TargetID.first + "-" + ConflictingTID->first << "' and '"
          << TargetID.first + "-" + ConflictingTID->second << "'";
      return reportError(createStringError(errc::invalid_argument, Msg.str()));
    }
  }

  if (CheckSection) {
    Expected<bool> Res = CheckBundledSection(BundlerConfig);
    if (!Res) {
      reportError(Res.takeError());
      return 1;
    }
    return !*Res;
  }

  // HIP uses clang-offload-bundler to bundle device-only compilation results
  // for multiple GPU archs, therefore allow no host target if all entries
  // are for HIP.
  BundlerConfig.AllowNoHost = HIPOnly;

  // Host triple is not really needed for unbundling operation, so do not
  // treat missing host triple as error if we do unbundling.
  if ((Unbundle && HostTargetNum > 1) ||
      (!Unbundle && HostTargetNum != 1 && !BundlerConfig.AllowNoHost)) {
    return reportError(createStringError(
        errc::invalid_argument,
        "expecting exactly one host target but got " + Twine(HostTargetNum)));
  }

  OffloadBundler Bundler(BundlerConfig);

  return doWork([&]() {
    if (Unbundle)
      return (BundlerConfig.FilesType == "a") ? Bundler.UnbundleArchive()
                                              : Bundler.UnbundleFiles();
    return Bundler.BundleFiles();
  });
}
