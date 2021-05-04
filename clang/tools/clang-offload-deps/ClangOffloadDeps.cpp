//===----------- clang-offload-deps/ClangOffloadDeps.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the clang-offload-deps tool. This tool is intended to be
/// used by the clang driver for offload linking with static offload libraries.
/// It takes linked host image as input and produces bitcode files, one per
/// offload target, containing references to symbols that must be defined in the
/// target images. Dependence bitcode file is then expected to be compiled to an
/// object by the driver using the appropriate offload target toolchain. This
/// dependence object is added to the target linker as input together with the
/// other inputs. References to the symbols in dependence object should ensure
/// that target linker pulls in necessary symbol definitions from the input
/// static libraries.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#ifndef NDEBUG
#include "llvm/IR/Verifier.h"
#endif // NDEBUG
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

#define SYMBOLS_SECTION_NAME ".tgtsym"

using namespace llvm;
using namespace llvm::object;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangOffloadDepsCategory("clang-offload-deps options");

static cl::list<std::string> Outputs("outputs", cl::CommaSeparated,
                                     cl::OneOrMore,
                                     cl::desc("[<output file>,...]"),
                                     cl::cat(ClangOffloadDepsCategory));
static cl::list<std::string>
    Targets("targets", cl::CommaSeparated, cl::OneOrMore,
            cl::desc("[<offload kind>-<target triple>,...]"),
            cl::cat(ClangOffloadDepsCategory));

static cl::opt<std::string> Input(cl::Positional, cl::Required,
                                  cl::desc("<input file>"),
                                  cl::cat(ClangOffloadDepsCategory));

/// Path to the current binary.
static std::string ToolPath;

static void reportError(Error E) {
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), ToolPath));
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  ToolPath = argv[0];

  cl::HideUnrelatedOptions(ClangOffloadDepsCategory);
  cl::SetVersionPrinter([](raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clang-offload-deps") << '\n';
  });
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool for creating dependence bitcode files for offload targets. "
      "Takes\nhost image as input and produces bitcode files, one per offload "
      "target, with\nreferences to symbols that must be defined in target "
      "images.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  // The number of output files and targets should match.
  if (Targets.size() != Outputs.size()) {
    reportError(
        createStringError(errc::invalid_argument,
                          "number of output files and targets should match"));
    return 1;
  }

  // Verify that given targets are valid. Each target string is expected to have
  // the following format
  //     <kind>-<triple>
  // where <kind> is host, openmp, hip, sycl or fpga,
  // and <triple> is an offload target triple.
  SmallVector<StringRef, 8u> Kinds(Targets.size());
  SmallVector<StringRef, 8u> Triples(Targets.size());
  for (unsigned I = 0; I < Targets.size(); ++I) {
    std::tie(Kinds[I], Triples[I]) = StringRef(Targets[I]).split('-');

    bool KindIsValid = StringSwitch<bool>(Kinds[I])
                           .Case("host", true)
                           .Case("openmp", true)
                           .Case("hip", true)
                           .Case("sycl", true)
                           .Case("fpga", true)
                           .Default(false);

    bool TripleIsValid = Triple(Triples[I]).getArch() != Triple::UnknownArch;

    if (!KindIsValid || !TripleIsValid) {
      SmallVector<char, 128u> Buf;
      raw_svector_ostream Msg(Buf);
      Msg << "invalid target '" << Targets[I] << "'";
      if (!KindIsValid)
        Msg << ", unknown offloading kind '" << Kinds[I] << "'";
      if (!TripleIsValid)
        Msg << ", unknown target triple '" << Triples[I] << "'";
      reportError(createStringError(errc::invalid_argument, Msg.str()));
      return 1;
    }
  }

  // Read input file. It should have one of the supported object file formats.
  Expected<OwningBinary<ObjectFile>> ObjectOrErr =
      ObjectFile::createObjectFile(Input);
  if (!ObjectOrErr) {
    reportError(ObjectOrErr.takeError());
    return 1;
  }

  // Then try to find a section in the input binary which contains offload
  // symbol names and parse section contents.
  DenseMap<StringRef, SmallDenseSet<StringRef>> Target2Symbols;
  for (SectionRef Section : ObjectOrErr->getBinary()->sections()) {
    // Look for the .tgtsym section in the binary.
    Expected<StringRef> NameOrErr = Section.getName();
    if (!NameOrErr) {
      reportError(NameOrErr.takeError());
      return 1;
    }
    if (*NameOrErr != SYMBOLS_SECTION_NAME)
      continue;

    // This is the section we are looking for, read symbol names from it.
    Expected<StringRef> DataOrErr = Section.getContents();
    if (!DataOrErr) {
      reportError(DataOrErr.takeError());
      return 1;
    }

    // Symbol names are prefixed by a target, and prefixed names are separated
    // by '\0' characters from each other. Find the names matching our list of
    // offload targets and insert them into the map.
    for (StringRef Symbol = DataOrErr.get(); !Symbol.empty();) {
      unsigned Len = strlen(Symbol.data());

      for (const std::string &Target : Targets) {
        std::string Prefix = Target + ".";
        if (Symbol.startswith(Prefix))
          Target2Symbols[Target].insert(
              Symbol.substr(Prefix.size(), Len - Prefix.size()));
      }

      Symbol = Symbol.drop_front(Len + 1u);
    }

    // Binary should not have more than one .tgtsym section.
    break;
  }

  LLVMContext Context;
  Type *Int8PtrTy = Type::getInt8PtrTy(Context);

  // Create bitcode file with the symbol names for each target and write it to
  // the output file.
  SmallVector<std::unique_ptr<ToolOutputFile>, 8u> Files;
  Files.reserve(Outputs.size());
  for (unsigned I = 0; I < Outputs.size(); ++I) {
    StringRef FileName = Outputs[I];

    Module Mod{"offload-deps", Context};
    Mod.setTargetTriple(Triples[I]);

    SmallVector<Constant *, 8u> Used;
    Used.reserve(Target2Symbols[Targets[I]].size());
    for (StringRef Symbol : Target2Symbols[Targets[I]])
      Used.push_back(ConstantExpr::getPointerBitCastOrAddrSpaceCast(
          Mod.getOrInsertGlobal(Symbol, Int8PtrTy), Int8PtrTy));

    if (!Used.empty()) {
      ArrayType *ArrayTy = ArrayType::get(Int8PtrTy, Used.size());

      // SYCL/SPIRV linking is done on LLVM IR inputs, so we can use special
      // global variable llvm.used to represent a reference to a symbol. But for
      // other targets we have to create a real reference since llvm.used may
      // not be representable in the object file.
      if (Kinds[I] == "sycl" || Triple(Triples[I]).isSPIR()) {
        auto *GV = new GlobalVariable(
            Mod, ArrayTy, false, GlobalValue::AppendingLinkage,
            ConstantArray::get(ArrayTy, Used), "llvm.used");
        GV->setSection("llvm.metadata");
      } else {
        auto *GV = new GlobalVariable(
            Mod, ArrayTy, false, GlobalValue::ExternalLinkage,
            ConstantArray::get(ArrayTy, Used), "offload.symbols");
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
        GV->setVisibility(GlobalValue::HiddenVisibility);
      }
    }

#ifndef NDEBUG
    if (verifyModule(Mod, &errs())) {
      reportError(createStringError(inconvertibleErrorCode(),
                                    "module verification error"));
      return 1;
    }
#endif // NDEBUG

    // Open output file.
    std::error_code EC;
    const auto &File = Files.emplace_back(
        std::make_unique<ToolOutputFile>(FileName, EC, sys::fs::OF_None));
    if (EC) {
      reportError(createFileError(FileName, EC));
      return 1;
    }

    // Write deps module to the output.
    WriteBitcodeToFile(Mod, File->os());
    if (File->os().has_error()) {
      reportError(createFileError(FileName, File->os().error()));
      return 1;
    }
  }

  // Everything is done, keep the output files.
  for (const auto &File : Files)
    File->keep();

  return 0;
}
