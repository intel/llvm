//===-- clang-linker-wrapper/ClangLinkerWrapper.cpp - wrapper over linker-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This tool works as a wrapper over a linking job. This tool is used to create
// linked device images for offloading. It scans the linker's input for embedded
// device offloading data stored in sections `.llvm.offloading` and extracts it
// as a temporary file. The extracted device files will then be passed to a
// device linking job to create a final device image.
//
//===---------------------------------------------------------------------===//

#include "clang/Basic/Cuda.h"
#include "clang/Basic/TargetID.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Frontend/Offloading/OffloadWrapper.h"
#include "llvm/Frontend/Offloading/SYCLOffloadWrapper.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Object/SYCLBIN.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include <atomic>
#include <optional>

#define COMPILE_OPTS "compile-opts"
#define LINK_OPTS "link-opts"

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::object;

// Various tools (e.g., llc and opt) duplicate this series of declarations for
// options related to passes and remarks.

static cl::opt<bool> RemarksWithHotness(
    "pass-remarks-with-hotness",
    cl::desc("With PGO, include profile count in optimization remarks"),
    cl::Hidden);

static cl::opt<std::optional<uint64_t>, false, remarks::HotnessThresholdParser>
    RemarksHotnessThreshold(
        "pass-remarks-hotness-threshold",
        cl::desc("Minimum profile count required for "
                 "an optimization remark to be output. "
                 "Use 'auto' to apply the threshold from profile summary."),
        cl::value_desc("N or 'auto'"), cl::init(0), cl::Hidden);

static cl::opt<std::string>
    RemarksFilename("pass-remarks-output",
                    cl::desc("Output filename for pass remarks"),
                    cl::value_desc("filename"));

static cl::opt<std::string>
    RemarksPasses("pass-remarks-filter",
                  cl::desc("Only record optimization remarks from passes whose "
                           "names match the given regular expression"),
                  cl::value_desc("regex"));

static cl::opt<std::string> RemarksFormat(
    "pass-remarks-format",
    cl::desc("The format used for serializing remarks (default: YAML)"),
    cl::value_desc("format"), cl::init("yaml"));

static cl::list<std::string>
    PassPlugins("load-pass-plugin",
                cl::desc("Load passes from plugin library"));

static cl::opt<std::string> PassPipeline(
    "passes",
    cl::desc(
        "A textual description of the pass pipeline. To have analysis passes "
        "available before a certain pass, add 'require<foo-analysis>'. "
        "'-passes' overrides the pass pipeline (but not all effects) from "
        "specifying '--opt-level=O?' (O2 is the default) to "
        "clang-linker-wrapper.  Be sure to include the corresponding "
        "'default<O?>' in '-passes'."));
static cl::alias PassPipeline2("p", cl::aliasopt(PassPipeline),
                               cl::desc("Alias for -passes"));

/// Path of the current binary.
static const char *LinkerExecutable;

/// Ssave intermediary results.
static bool SaveTemps = false;

/// Print arguments without executing.
static bool DryRun = false;

/// Print verbose output.
static bool Verbose = false;

/// Filename of the executable being created.
static StringRef ExecutableName;

/// Binary path for the CUDA installation.
static std::string CudaBinaryPath;

/// Mutex lock to protect writes to shared TempFiles in parallel.
static std::mutex TempFilesMutex;

/// Temporary files created by the linker wrapper.
static std::list<SmallString<128>> TempFiles;

/// Codegen flags for LTO backend.
static codegen::RegisterCodeGenFlags CodeGenFlags;

static std::optional<llvm::module_split::IRSplitMode> SYCLModuleSplitMode;

static bool UseSYCLPostLinkTool;

static bool OutputSYCLBIN = false;

static SYCLBIN::BundleState SYCLBINState = SYCLBIN::BundleState::Input;

static SmallString<128> OffloadImageDumpDir;

using OffloadingImage = OffloadBinary::OffloadingImage;

namespace llvm {
// Provide DenseMapInfo so that OffloadKind can be used in a DenseMap.
template <> struct DenseMapInfo<OffloadKind> {
  static inline OffloadKind getEmptyKey() { return OFK_LAST; }
  static inline OffloadKind getTombstoneKey() {
    return static_cast<OffloadKind>(OFK_LAST + 1);
  }
  static unsigned getHashValue(const OffloadKind &Val) { return Val; }

  static bool isEqual(const OffloadKind &LHS, const OffloadKind &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

namespace {
using std::error_code;

/// Must not overlap with llvm::opt::DriverFlag.
enum WrapperFlags {
  WrapperOnlyOption = (1 << 4), // Options only used by the linker wrapper.
  DeviceOnlyOption = (1 << 5),  // Options only used for device linking.
};

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "LinkerWrapperOpts.inc"
  LastOption
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "LinkerWrapperOpts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "LinkerWrapperOpts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "LinkerWrapperOpts.inc"
#undef OPTION
};

class WrapperOptTable : public opt::GenericOptTable {
public:
  WrapperOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};

const OptTable &getOptTable() {
  static const WrapperOptTable *Table = []() {
    auto Result = std::make_unique<WrapperOptTable>();
    return Result.release();
  }();
  return *Table;
}

void printCommands(ArrayRef<StringRef> CmdArgs) {
  if (CmdArgs.empty())
    return;

  llvm::errs() << " \"" << CmdArgs.front() << "\" ";
  for (auto IC = std::next(CmdArgs.begin()), IE = CmdArgs.end(); IC != IE; ++IC)
    llvm::errs() << *IC << (std::next(IC) != IE ? " " : "\n");
}

[[noreturn]] void reportError(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E),
                        WithColor::error(errs(), LinkerExecutable));
  exit(EXIT_FAILURE);
}

/// Create an extra user-specified \p OffloadFile.
/// TODO: We should find a way to wrap these as libraries instead.
Expected<OffloadFile> getInputBitcodeLibrary(StringRef Input) {
  auto [Device, Path] = StringRef(Input).split('=');
  auto [String, Arch] = Device.rsplit('-');
  auto [Kind, Triple] = String.split('-');

  llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> ImageOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(Path);
  if (std::error_code EC = ImageOrError.getError())
    return createFileError(Path, EC);

  OffloadingImage Image{};
  Image.TheImageKind = IMG_Bitcode;
  Image.TheOffloadKind = getOffloadKind(Kind);
  Image.StringData["triple"] = Triple;
  Image.StringData["arch"] = Arch;
  Image.Image = std::move(*ImageOrError);

  std::unique_ptr<MemoryBuffer> Binary = MemoryBuffer::getMemBufferCopy(
      OffloadBinary::write(Image), Image.Image->getBufferIdentifier());
  auto NewBinaryOrErr = OffloadBinary::create(*Binary);
  if (!NewBinaryOrErr)
    return NewBinaryOrErr.takeError();
  return OffloadFile(std::move(*NewBinaryOrErr), std::move(Binary));
}

std::string getMainExecutable(const char *Name) {
  void *Ptr = (void *)(intptr_t)&getMainExecutable;
  auto COWPath = sys::fs::getMainExecutable(Name, Ptr);
  return sys::path::parent_path(COWPath).str();
}

/// Get a temporary filename suitable for output.
Expected<StringRef> createOutputFile(const Twine &Prefix, StringRef Extension) {
  std::scoped_lock<decltype(TempFilesMutex)> Lock(TempFilesMutex);
  SmallString<128> OutputFile;
  if (SaveTemps) {
    // Generate a unique path name without creating a file
    sys::fs::createUniquePath(Prefix + "-%%%%%%." + Extension, OutputFile,
                              /*MakeAbsolute=*/false);
  } else {
    if (std::error_code EC =
            sys::fs::createTemporaryFile(Prefix, Extension, OutputFile))
      return createFileError(OutputFile, EC);
  }

  TempFiles.emplace_back(std::move(OutputFile));
  return TempFiles.back();
}

Error containerizeRawImage(std::unique_ptr<MemoryBuffer> &Img, OffloadKind Kind,
                           const ArgList &Args) {
  llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  if (Kind != OFK_OpenMP || !Triple.isSPIRV() ||
      Triple.getVendor() != llvm::Triple::Intel)
    return Error::success();
  return offloading::intel::containerizeOpenMPSPIRVImage(Img);
}

// TODO: Remove HasSYCLOffloadKind dependence when aligning with community code.
Expected<StringRef> writeOffloadFile(const OffloadFile &File,
                                     bool HasSYCLOffloadKind = false) {
  const OffloadBinary &Binary = *File.getBinary();

  StringRef Prefix =
      sys::path::stem(Binary.getMemoryBufferRef().getBufferIdentifier());
  StringRef Suffix = getImageKindName(Binary.getImageKind());

  auto TempFileOrErr = createOutputFile(
      Prefix + "-" + Binary.getTriple() + "-" + Binary.getArch(),
      HasSYCLOffloadKind ? getImageKindName(Binary.getImageKind()) : "o");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
      FileOutputBuffer::create(*TempFileOrErr, Binary.getImage().size());
  if (!OutputOrErr)
    return OutputOrErr.takeError();
  std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
  llvm::copy(Binary.getImage(), Output->getBufferStart());
  if (Error E = Output->commit())
    return std::move(E);

  return *TempFileOrErr;
}

/// Execute the command \p ExecutablePath with the arguments \p Args.
Error executeCommands(StringRef ExecutablePath, ArrayRef<StringRef> Args) {
  if (Verbose || DryRun)
    printCommands(Args);

  if (!DryRun)
    if (sys::ExecuteAndWait(ExecutablePath, Args))
      return createStringError(
          "'%s' failed", sys::path::filename(ExecutablePath).str().c_str());
  return Error::success();
}

Expected<std::string> findProgram(StringRef Name, ArrayRef<StringRef> Paths) {

  ErrorOr<std::string> Path = sys::findProgramByName(Name, Paths);
  if (!Path)
    Path = sys::findProgramByName(Name);
  if (!Path && DryRun)
    return Name.str();
  if (!Path)
    return createStringError(Path.getError(),
                             "Unable to find '" + Name + "' in path");
  return *Path;
}

bool linkerSupportsLTO(const ArgList &Args) {
  llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  return Triple.isNVPTX() || Triple.isAMDGPU() ||
         (!Triple.isGPU() &&
          Args.getLastArgValue(OPT_linker_path_EQ).ends_with("lld"));
}

/// Returns the hashed value for a constant string.
std::string getHash(StringRef Str) {
  llvm::MD5 Hasher;
  llvm::MD5::MD5Result Hash;
  Hasher.update(Str);
  Hasher.final(Hash);
  return llvm::utohexstr(Hash.low(), /*LowerCase=*/true);
}

/// Renames offloading entry sections in a relocatable link so they do not
/// conflict with a later link job.
Error relocateOffloadSection(const ArgList &Args, StringRef Output) {
  llvm::Triple Triple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple()));
  if (Triple.isOSWindows())
    return createStringError(
        "Relocatable linking is not supported on COFF targets");

  Expected<std::string> ObjcopyPath =
      findProgram("llvm-objcopy", {getMainExecutable("llvm-objcopy")});
  if (!ObjcopyPath)
    return ObjcopyPath.takeError();

  // Use the linker output file to get a unique hash. This creates a unique
  // identifier to rename the sections to that is deterministic to the contents.
  auto BufferOrErr = DryRun ? MemoryBuffer::getMemBuffer("")
                            : MemoryBuffer::getFileOrSTDIN(Output);
  if (!BufferOrErr)
    return createStringError("Failed to open %s", Output.str().c_str());
  std::string Suffix = "_" + getHash((*BufferOrErr)->getBuffer());

  SmallVector<StringRef> ObjcopyArgs = {
      *ObjcopyPath,
      Output,
  };

  // Remove the old .llvm.offloading section to prevent further linking.
  ObjcopyArgs.emplace_back("--remove-section");
  ObjcopyArgs.emplace_back(".llvm.offloading");
  StringRef Prefix = "llvm";
  auto Section = (Prefix + "_offload_entries").str();
  // Rename the offloading entires to make them private to this link unit.
  ObjcopyArgs.emplace_back("--rename-section");
  ObjcopyArgs.emplace_back(
      Args.MakeArgString(Section + "=" + Section + Suffix));

  // Rename the __start_ / __stop_ symbols appropriately to iterate over the
  // newly renamed section containing the offloading entries.
  ObjcopyArgs.emplace_back("--redefine-sym");
  ObjcopyArgs.emplace_back(Args.MakeArgString("__start_" + Section + "=" +
                                              "__start_" + Section + Suffix));
  ObjcopyArgs.emplace_back("--redefine-sym");
  ObjcopyArgs.emplace_back(Args.MakeArgString("__stop_" + Section + "=" +
                                              "__stop_" + Section + Suffix));

  if (Error Err = executeCommands(*ObjcopyPath, ObjcopyArgs))
    return Err;

  return Error::success();
}

/// Runs the wrapped linker job with the newly created input.
Error runLinker(ArrayRef<StringRef> Files, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("Execute host linker");

  // Render the linker arguments and add the newly created image. We add it
  // after the output file to ensure it is linked with the correct libraries.
  StringRef LinkerPath = Args.getLastArgValue(OPT_linker_path_EQ);
  if (LinkerPath.empty())
    return createStringError("linker path missing, must pass 'linker-path'");
  ArgStringList NewLinkerArgs;
  for (const opt::Arg *Arg : Args) {
    // Do not forward arguments only intended for the linker wrapper.
    if (Arg->getOption().hasFlag(WrapperOnlyOption))
      continue;

    Arg->render(Args, NewLinkerArgs);
    if (Arg->getOption().matches(OPT_o) || Arg->getOption().matches(OPT_out))
      llvm::transform(Files, std::back_inserter(NewLinkerArgs),
                      [&](StringRef Arg) { return Args.MakeArgString(Arg); });
  }

  SmallVector<StringRef> LinkerArgs({LinkerPath});
  for (StringRef Arg : NewLinkerArgs)
    LinkerArgs.push_back(Arg);
  if (Error Err = executeCommands(LinkerPath, LinkerArgs))
    return Err;

  if (Args.hasArg(OPT_relocatable))
    return relocateOffloadSection(Args, ExecutableName);

  return Error::success();
}

void printVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-linker-wrapper") << '\n';
}

namespace nvptx {
Expected<StringRef>
fatbinary(ArrayRef<std::pair<StringRef, StringRef>> InputFiles,
          const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("NVPTX fatbinary");
  // NVPTX uses the fatbinary program to bundle the linked images.
  Expected<std::string> FatBinaryPath =
      findProgram("fatbinary", {CudaBinaryPath + "/bin"});
  if (!FatBinaryPath)
    return FatBinaryPath.takeError();

  llvm::Triple Triple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple()));

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "fatbin");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*FatBinaryPath);
  CmdArgs.push_back(Triple.isArch64Bit() ? "-64" : "-32");
  CmdArgs.push_back("--create");
  CmdArgs.push_back(*TempFileOrErr);
  for (const auto &[File, Arch] : InputFiles)
    CmdArgs.push_back(
        Args.MakeArgString("--image=profile=" + Arch + ",file=" + File));

  if (Error Err = executeCommands(*FatBinaryPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}

// ptxas binary
Expected<StringRef> ptxas(StringRef InputFile, const ArgList &Args,
                          StringRef Arch) {
  llvm::TimeTraceScope TimeScope("NVPTX ptxas");
  // NVPTX uses the ptxas program to process assembly files.
  Expected<std::string> PtxasPath =
      findProgram("ptxas", {CudaBinaryPath + "/bin"});
  if (!PtxasPath)
    return PtxasPath.takeError();

  llvm::Triple Triple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple()));

  // Create a new file to write the output to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "cubin");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*PtxasPath);
  CmdArgs.push_back(Triple.isArch64Bit() ? "-m64" : "-m32");
  // Pass -v to ptxas if it was passed to the driver.
  CmdArgs.push_back("--gpu-name");
  CmdArgs.push_back(Arch);
  CmdArgs.push_back("--output-file");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back(InputFile);
  if (Error Err = executeCommands(*PtxasPath, CmdArgs))
    return std::move(Err);
  return *TempFileOrErr;
}
} // namespace nvptx

namespace amdgcn {
Expected<StringRef>
fatbinary(ArrayRef<std::pair<StringRef, StringRef>> InputFiles,
          const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("AMDGPU Fatbinary");

  // AMDGPU uses the clang-offload-bundler to bundle the linked images.
  Expected<std::string> OffloadBundlerPath = findProgram(
      "clang-offload-bundler", {getMainExecutable("clang-offload-bundler")});
  if (!OffloadBundlerPath)
    return OffloadBundlerPath.takeError();

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "hipfb");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*OffloadBundlerPath);
  CmdArgs.push_back("-type=o");
  CmdArgs.push_back("-bundle-align=4096");

  if (Args.hasArg(OPT_compress))
    CmdArgs.push_back("-compress");
  if (auto *Arg = Args.getLastArg(OPT_compression_level_eq))
    CmdArgs.push_back(
        Args.MakeArgString(Twine("-compression-level=") + Arg->getValue()));

  SmallVector<StringRef> Targets = {"-targets=host-x86_64-unknown-linux-gnu"};
  for (const auto &[File, Arch] : InputFiles)
    Targets.push_back(Saver.save("hip-amdgcn-amd-amdhsa--" + Arch));
  CmdArgs.push_back(Saver.save(llvm::join(Targets, ",")));

#ifdef _WIN32
  CmdArgs.push_back("-input=NUL");
#else
  CmdArgs.push_back("-input=/dev/null");
#endif
  for (const auto &[File, Arch] : InputFiles)
    CmdArgs.push_back(Saver.save("-input=" + File));

  CmdArgs.push_back(Saver.save("-output=" + *TempFileOrErr));

  if (Error Err = executeCommands(*OffloadBundlerPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}
} // namespace amdgcn

namespace sycl {
// This utility function is used to gather all SYCL device library files that
// will be linked with input device files.
// The list of files and its location are passed from driver.
static Error getSYCLDeviceLibs(SmallVector<std::string, 16> &DeviceLibFiles,
                               const ArgList &Args) {
  StringRef SYCLDeviceLibLoc("");
  if (Arg *A = Args.getLastArg(OPT_sycl_device_library_location_EQ))
    SYCLDeviceLibLoc = A->getValue();
  if (Arg *A = Args.getLastArg(OPT_sycl_device_lib_EQ)) {
    if (A->getValues().size() == 0)
      return createStringError(
          inconvertibleErrorCode(),
          "Number of device library files cannot be zero.");
    for (StringRef Val : A->getValues()) {
      SmallString<128> LibName(SYCLDeviceLibLoc);
      llvm::sys::path::append(LibName, Val);
      if (llvm::sys::fs::exists(LibName))
        DeviceLibFiles.push_back(std::string(LibName));
      else
        return createStringError(inconvertibleErrorCode(),
                                 std::string(LibName) +
                                     " SYCL device library file is not found.");
    }
  }
  return Error::success();
}

/// This routine is used to convert SPIR-V input files into LLVM IR files.
/// 'llvm-spirv -r' command is used for this purpose.
/// If input is not a SPIR-V file, then the original file is returned.
/// TODO: Add a check to identify SPIR-V files and exit early if the input is
/// not a SPIR-V file.
/// 'Filename' is the input file that could be a SPIR-V file.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// llvm-spirv tool.
static Expected<StringRef> convertSPIRVToIR(StringRef Filename,
                                            const ArgList &Args) {
  Expected<std::string> SPIRVToIRWrapperPath = findProgram(
      "spirv-to-ir-wrapper", {getMainExecutable("spirv-to-ir-wrapper")});
  if (!SPIRVToIRWrapperPath)
    return SPIRVToIRWrapperPath.takeError();

  // Create a new file to write the converted file to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "bc");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*SPIRVToIRWrapperPath);
  CmdArgs.push_back(Filename);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back("--llvm-spirv-opts");
  CmdArgs.push_back("--spirv-preserve-auxdata --spirv-target-env=SPV-IR "
                    "--spirv-builtin-format=global");
  if (Error Err = executeCommands(*SPIRVToIRWrapperPath, CmdArgs))
    return std::move(Err);
  return *TempFileOrErr;
}

/// Add any sycl-post-link options that rely on a specific Triple in addition
/// to user supplied options.
/// NOTE: Any changes made here should be reflected in the similarly named
/// function in clang/lib/Driver/ToolChains/Clang.cpp.
static void
getTripleBasedSYCLPostLinkOpts(const ArgList &Args,
                               SmallVector<StringRef, 8> &PostLinkArgs,
                               const llvm::Triple Triple) {
  const llvm::Triple HostTriple(Args.getLastArgValue(OPT_host_triple_EQ));
  bool SpecConstsSupported = (!Triple.isNVPTX() && !Triple.isAMDGCN() &&
                              !Triple.isSPIRAOT() && !Triple.isNativeCPU());
  if (SpecConstsSupported)
    PostLinkArgs.push_back("-spec-const=native");
  else
    PostLinkArgs.push_back("-spec-const=emulation");

  // TODO: If we ever pass -ir-output-only based on the triple,
  // make sure we don't pass -properties.
  PostLinkArgs.push_back("-properties");

  // See if device code splitting is already requested. If not requested, then
  // set -split=auto for non-FPGA targets.
  bool NoSplit = true;
  for (auto Arg : PostLinkArgs)
    if (Arg.contains("-split=")) {
      NoSplit = false;
      break;
    }
  if (NoSplit && (Triple.getSubArch() != llvm::Triple::SPIRSubArch_fpga))
    PostLinkArgs.push_back("-split=auto");

  // On Intel targets we don't need non-kernel functions as entry points,
  // because it only increases amount of code for device compiler to handle,
  // without any actual benefits.
  // TODO: Try to extend this feature for non-Intel GPUs.
  if (!Args.hasFlag(OPT_no_sycl_remove_unused_external_funcs,
                    OPT_sycl_remove_unused_external_funcs, false) &&
      !Args.hasArg(OPT_sycl_allow_device_image_dependencies) &&
      !Triple.isNVPTX() && !Triple.isAMDGPU())
    PostLinkArgs.push_back("-emit-only-kernels-as-entry-points");

  if (!Triple.isAMDGCN())
    PostLinkArgs.push_back("-emit-param-info");
  // Enable program metadata
  if (Triple.isNVPTX() || Triple.isAMDGCN() || Triple.isNativeCPU())
    PostLinkArgs.push_back("-emit-program-metadata");

  bool SplitEsimdByDefault = Triple.isSPIROrSPIRV();
  bool SplitEsimd =
      Args.hasFlag(OPT_sycl_device_code_split_esimd,
                   OPT_no_sycl_device_code_split_esimd, SplitEsimdByDefault);
  if (!Args.hasArg(OPT_sycl_thin_lto))
    PostLinkArgs.push_back("-symbols");
  // Emit kernel names if we are producing SYCLBIN.
  if (Args.hasArg(OPT_syclbin_EQ))
    PostLinkArgs.push_back("-emit-kernel-names");
  // Specialization constant info generation is mandatory -
  // add options unconditionally
  PostLinkArgs.push_back("-emit-exported-symbols");
  PostLinkArgs.push_back("-emit-imported-symbols");
  if (SplitEsimd)
    PostLinkArgs.push_back("-split-esimd");
  PostLinkArgs.push_back("-lower-esimd");

  bool IsAOT = Triple.isNVPTX() || Triple.isAMDGCN() || Triple.isSPIRAOT();
  if (Args.hasFlag(OPT_sycl_add_default_spec_consts_image,
                   OPT_no_sycl_add_default_spec_consts_image, false) &&
      IsAOT)
    PostLinkArgs.push_back("-generate-device-image-default-spec-consts");
}

/// Run sycl-post-link tool for SYCL offloading.
/// 'InputFiles' is the list of input LLVM IR files.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// sycl-post-link tool.
static Expected<std::vector<module_split::SplitModule>>
runSYCLPostLinkTool(ArrayRef<StringRef> InputFiles, const ArgList &Args) {
  Expected<std::string> SYCLPostLinkPath =
      findProgram("sycl-post-link", {getMainExecutable("sycl-post-link")});
  if (!SYCLPostLinkPath)
    return SYCLPostLinkPath.takeError();

  // Create a new file to write the output of sycl-post-link to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "table");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*SYCLPostLinkPath);
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  Arg *SYCLDeviceLibLoc = Args.getLastArg(OPT_sycl_device_library_location_EQ);
  if (SYCLDeviceLibLoc && !Triple.isSPIRAOT()) {
    std::string SYCLDeviceLibLocParam = SYCLDeviceLibLoc->getValue();
    std::string BF16DeviceLibLoc =
        SYCLDeviceLibLocParam + "/libsycl-native-bfloat16.bc";
    if (llvm::sys::fs::exists(BF16DeviceLibLoc)) {
      SYCLDeviceLibLocParam = "--device-lib-dir=" + SYCLDeviceLibLocParam;
      CmdArgs.push_back(Args.MakeArgString(StringRef(SYCLDeviceLibLocParam)));
    }
  }
  getTripleBasedSYCLPostLinkOpts(Args, CmdArgs, Triple);
  StringRef SYCLPostLinkOptions;
  if (Arg *A = Args.getLastArg(OPT_sycl_post_link_options_EQ))
    SYCLPostLinkOptions = A->getValue();
  SYCLPostLinkOptions.split(CmdArgs, " ", /* MaxSplit = */ -1,
                            /* KeepEmpty = */ false);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);
  for (auto &File : InputFiles)
    CmdArgs.push_back(File);
  if (Error Err = executeCommands(*SYCLPostLinkPath, CmdArgs))
    return std::move(Err);

  if (DryRun) {
    // In DryRun we need a dummy entry in order to continue the whole pipeline.
    auto ImageFileOrErr = createOutputFile(
        sys::path::filename(ExecutableName) + ".sycl.split.image", "bc");
    if (!ImageFileOrErr)
      return ImageFileOrErr.takeError();

    std::vector Modules = {module_split::SplitModule(
        *ImageFileOrErr, util::PropertySetRegistry(), "")};
    return Modules;
  }

  return llvm::module_split::parseSplitModulesFromFile(*TempFileOrErr);
}

/// Invokes SYCL Split library for SYCL offloading.
///
/// \param InputFiles the list of input LLVM IR files.
/// \param Args Encompasses all arguments for linking and wrapping device code.
///  It will be parsed to generate options required to be passed to SYCL split
///  library.
/// \param Mode The splitting mode.
/// \returns The vector of split modules.
static Expected<std::vector<module_split::SplitModule>>
runSYCLSplitLibrary(ArrayRef<StringRef> InputFiles, const ArgList &Args,
                    module_split::IRSplitMode Mode) {
  std::vector<module_split::SplitModule> SplitModules;
  if (DryRun) {
    auto OutputFileOrErr = createOutputFile(
        sys::path::filename(ExecutableName) + ".sycl.split.image", "bc");
    if (!OutputFileOrErr)
      return OutputFileOrErr.takeError();

    StringRef OutputFilePath = *OutputFileOrErr;
    auto InputFilesStr = llvm::join(InputFiles.begin(), InputFiles.end(), ",");
    errs() << formatv("sycl-module-split: input: {0}, output: {1}\n",
                      InputFilesStr, OutputFilePath);
    SplitModules.emplace_back(OutputFilePath, util::PropertySetRegistry(), "");
    return SplitModules;
  }

  llvm::module_split::ModuleSplitterSettings Settings;
  Settings.Mode = Mode;
  Settings.OutputPrefix = "";

  for (StringRef InputFile : InputFiles) {
    SMDiagnostic Err;
    LLVMContext C;
    std::unique_ptr<Module> M = parseIRFile(InputFile, Err, C);
    if (!M)
      return createStringError(inconvertibleErrorCode(), Err.getMessage());

    auto SplitModulesOrErr =
        module_split::splitSYCLModule(std::move(M), Settings);
    if (!SplitModulesOrErr)
      return SplitModulesOrErr.takeError();

    auto &NewSplitModules = *SplitModulesOrErr;
    SplitModules.insert(SplitModules.end(), NewSplitModules.begin(),
                        NewSplitModules.end());
  }

  if (Verbose) {
    auto InputFilesStr = llvm::join(InputFiles.begin(), InputFiles.end(), ",");
    std::string SplitOutputFilesStr;
    for (size_t I = 0, E = SplitModules.size(); I != E; ++I) {
      if (I > 0)
        SplitOutputFilesStr += ',';

      SplitOutputFilesStr += SplitModules[I].ModuleFilePath;
    }

    errs() << formatv("sycl-module-split: input: {0}, output: {1}\n",
                      InputFilesStr, SplitOutputFilesStr);
  }

  return SplitModules;
}

/// Add any llvm-spirv option that relies on a specific Triple in addition
/// to user supplied options.
/// NOTE: Any changes made here should be reflected in the similarly named
/// function in clang/lib/Driver/ToolChains/Clang.cpp.
static void
getTripleBasedSPIRVTransOpts(const ArgList &Args,
                             SmallVector<StringRef, 8> &TranslatorArgs,
                             const llvm::Triple Triple) {
  bool IsCPU = Triple.isSPIR() &&
               Triple.getSubArch() == llvm::Triple::SPIRSubArch_x86_64;
  TranslatorArgs.push_back("-spirv-debug-info-version=nonsemantic-shader-200");
  std::string UnknownIntrinsics("-spirv-allow-unknown-intrinsics=llvm.genx.");
  if (IsCPU)
    UnknownIntrinsics += ",llvm.fpbuiltin";
  TranslatorArgs.push_back(Args.MakeArgString(UnknownIntrinsics));

  // Disable all the extensions by default
  std::string ExtArg("-spirv-ext=-all");
  std::string DefaultExtArg =
      ",+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max"
      ",+SPV_KHR_no_integer_wrap_decoration,+SPV_KHR_float_controls"
      ",+SPV_KHR_expect_assume,+SPV_KHR_linkonce_odr";
  std::string INTELExtArg =
      ",+SPV_INTEL_subgroups,+SPV_INTEL_media_block_io"
      ",+SPV_INTEL_device_side_avc_motion_estimation"
      ",+SPV_INTEL_fpga_loop_controls,+SPV_INTEL_unstructured_loop_controls"
      ",+SPV_INTEL_fpga_reg,+SPV_INTEL_blocking_pipes"
      ",+SPV_INTEL_function_pointers,+SPV_INTEL_kernel_attributes"
      ",+SPV_INTEL_io_pipes,+SPV_INTEL_inline_assembly"
      ",+SPV_INTEL_arbitrary_precision_integers"
      ",+SPV_INTEL_float_controls2,+SPV_INTEL_vector_compute"
      ",+SPV_INTEL_fast_composite"
      ",+SPV_INTEL_arbitrary_precision_fixed_point"
      ",+SPV_INTEL_arbitrary_precision_floating_point"
      ",+SPV_INTEL_variable_length_array,+SPV_INTEL_fp_fast_math_mode"
      ",+SPV_INTEL_long_composites"
      ",+SPV_INTEL_arithmetic_fence"
      ",+SPV_INTEL_global_variable_decorations"
      ",+SPV_INTEL_cache_controls"
      ",+SPV_INTEL_fpga_buffer_location"
      ",+SPV_INTEL_fpga_argument_interfaces"
      ",+SPV_INTEL_fpga_invocation_pipelining_attributes"
      ",+SPV_INTEL_fpga_latency_control"
      ",+SPV_KHR_shader_clock"
      ",+SPV_INTEL_bindless_images"
      ",+SPV_INTEL_task_sequence";
  ExtArg = ExtArg + DefaultExtArg + INTELExtArg;
  ExtArg += ",+SPV_INTEL_bfloat16_conversion"
            ",+SPV_INTEL_joint_matrix"
            ",+SPV_INTEL_hw_thread_queries"
            ",+SPV_KHR_uniform_group_instructions"
            ",+SPV_INTEL_masked_gather_scatter"
            ",+SPV_INTEL_tensor_float32_conversion"
            ",+SPV_INTEL_optnone"
            ",+SPV_KHR_non_semantic_info"
            ",+SPV_KHR_cooperative_matrix"
            ",+SPV_EXT_shader_atomic_float16_add"
            ",+SPV_INTEL_fp_max_error";
  TranslatorArgs.push_back(Args.MakeArgString(ExtArg));
}

/// Run LLVM to SPIR-V translation.
/// Converts 'File' from LLVM bitcode to SPIR-V format using llvm-spirv tool.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// llvm-spirv tool.
static Expected<StringRef> runLLVMToSPIRVTranslation(StringRef File,
                                                     const ArgList &Args) {
  Expected<std::string> LLVMToSPIRVPath =
      findProgram("llvm-spirv", {getMainExecutable("llvm-spirv")});
  if (!LLVMToSPIRVPath)
    return LLVMToSPIRVPath.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*LLVMToSPIRVPath);
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  getTripleBasedSPIRVTransOpts(Args, CmdArgs, Triple);
  StringRef LLVMToSPIRVOptions;
  if (Arg *A = Args.getLastArg(OPT_llvm_spirv_options_EQ))
    LLVMToSPIRVOptions = A->getValue();
  LLVMToSPIRVOptions.split(CmdArgs, " ", /* MaxSplit = */ -1,
                           /* KeepEmpty = */ false);
  CmdArgs.push_back("-o");

  // Create a new file to write the translated file to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "spv");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back(File);
  if (Error Err = executeCommands(*LLVMToSPIRVPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}

/// Adds all AOT backend options required for SYCL AOT compilation step to
/// 'CmdArgs'.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate backend options required to be passed
/// into the SYCL AOT compilation step.
/// IsCPU is a bool used to direct option generation. If IsCPU is false, then
/// options are generated for AOT compilation targeting Intel GPUs.
static void addBackendOptions(const ArgList &Args,
                              SmallVector<StringRef, 8> &CmdArgs, bool IsCPU) {
  StringRef OptC =
      Args.getLastArgValue(OPT_sycl_backend_compile_options_from_image_EQ);
  OptC.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  StringRef OptL =
      Args.getLastArgValue(OPT_sycl_backend_link_options_from_image_EQ);
  OptL.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  StringRef OptTool = (IsCPU) ? Args.getLastArgValue(OPT_cpu_tool_arg_EQ)
                              : Args.getLastArgValue(OPT_gpu_tool_arg_EQ);
  OptTool.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  return;
}

/// Run AOT compilation for Intel CPU.
/// Calls opencl-aot tool to generate device code for Intel CPU backend.
/// 'InputFile' is the input SPIR-V file.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// SYCL AOT compilation step.
static Expected<StringRef> runAOTCompileIntelCPU(StringRef InputFile,
                                                 const ArgList &Args) {
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  SmallVector<StringRef, 8> CmdArgs;
  Expected<std::string> OpenCLAOTPath =
      findProgram("opencl-aot", {getMainExecutable("opencl-aot")});
  if (!OpenCLAOTPath)
    return OpenCLAOTPath.takeError();

  CmdArgs.push_back(*OpenCLAOTPath);
  CmdArgs.push_back("--device=cpu");
  addBackendOptions(Args, CmdArgs, /* IsCPU */ true);
  // Create a new file to write the translated file to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "out");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back(InputFile);
  if (Error Err = executeCommands(*OpenCLAOTPath, CmdArgs))
    return std::move(Err);
  return *TempFileOrErr;
}

/// Run AOT compilation for Intel GPU
/// Calls ocloc tool to generate device code for Intel GPU backend.
/// 'InputFile' is the input SPIR-V file.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// SYCL AOT compilation step.
static Expected<StringRef> runAOTCompileIntelGPU(StringRef InputFile,
                                                 const ArgList &Args) {
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  StringRef Arch(Args.getLastArgValue(OPT_arch_EQ));
  SmallVector<StringRef, 8> CmdArgs;
  Expected<std::string> OclocPath =
      findProgram("ocloc", {getMainExecutable("ocloc")});
  if (!OclocPath)
    return OclocPath.takeError();

  CmdArgs.push_back(*OclocPath);
  // The next line prevents ocloc from modifying the image name
  CmdArgs.push_back("-output_no_suffix");
  CmdArgs.push_back("-spirv_input");
  if (!Arch.empty()) {
    CmdArgs.push_back("-device");
    CmdArgs.push_back(Arch);
  }
  addBackendOptions(Args, CmdArgs, /* IsCPU */ false);
  // Create a new file to write the translated file to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "out");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();
  CmdArgs.push_back("-output");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back("-file");
  CmdArgs.push_back(InputFile);
  if (Error Err = executeCommands(*OclocPath, CmdArgs))
    return std::move(Err);
  return *TempFileOrErr;
}

/// Run AOT compilation for Intel CPU/GPU.
/// 'InputFile' is the input SPIR-V file.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// SYCL AOT compilation step.
static Expected<StringRef> runAOTCompile(StringRef InputFile,
                                         const ArgList &Args) {
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  if (Triple.isSPIRAOT()) {
    if (Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen)
      return runAOTCompileIntelGPU(InputFile, Args);
    if (Triple.getSubArch() == llvm::Triple::SPIRSubArch_x86_64)
      return runAOTCompileIntelCPU(InputFile, Args);
  }
  return createStringError(inconvertibleErrorCode(),
                           "Unsupported SYCL Triple and Arch");
}

/// Reads device images from the given \p InputFile and wraps them
/// in one LLVM IR Module as a constant data.
///
/// \returns A path to the LLVM Module that contains wrapped images.
Expected<StringRef>
wrapSYCLBinariesFromFile(std::vector<module_split::SplitModule> &SplitModules,
                         const ArgList &Args, bool IsEmbeddedIR) {
  auto OutputFileOrErr = createOutputFile(
      sys::path::filename(ExecutableName) + ".sycl.image.wrapper", "bc");
  if (!OutputFileOrErr)
    return OutputFileOrErr.takeError();

  StringRef OutputFilePath = *OutputFileOrErr;
  if (Verbose || DryRun) {
    std::string InputFiles;
    for (size_t I = 0, E = SplitModules.size(); I != E; ++I) {
      InputFiles += SplitModules[I].ModuleFilePath;
      if (I + 1 < E)
        InputFiles += ',';
    }

    errs() << formatv(" offload-wrapper: input: {0}, output: {1}\n", InputFiles,
                      OutputFilePath);
    if (DryRun)
      return OutputFilePath;
  }

  StringRef Target = Args.getLastArgValue(OPT_triple_EQ);
  if (Target.empty())
    return createStringError(
        inconvertibleErrorCode(),
        "can't wrap SYCL image. -triple argument is missed.");

  SmallVector<llvm::offloading::SYCLImage> Images;
  // SYCL runtime currently works for spir64 target triple and not for
  // spir64-unknown-unknown/spirv64-unknown-unknown/spirv64.
  // TODO: Fix SYCL runtime to accept other triples
  llvm::Triple T(Target);
  std::string EmbeddedIRTarget("llvm_");
  EmbeddedIRTarget.append(T.getArchName());
  StringRef RegularTarget(T.getArchName());
  if (RegularTarget == "spirv64")
    RegularTarget = "spir64";

  for (auto &SI : SplitModules) {
    if (!OffloadImageDumpDir.empty()) {
      StringRef CopyFrom = SI.ModuleFilePath;
      SmallString<128> CopyTo = OffloadImageDumpDir;
      StringRef Filename = sys::path::filename(CopyFrom);
      CopyTo.append(Filename);
      std::error_code EC = sys::fs::copy_file(CopyFrom, CopyTo);
      if (EC)
        return createStringError(EC, formatv("failed to copy file. From: "
                                             "{0} to: {1}, error_code: {2}",
                                             CopyFrom, CopyTo, EC.value()));
    }

    auto MBOrDesc = MemoryBuffer::getFile(SI.ModuleFilePath);
    if (!MBOrDesc)
      return createFileError(SI.ModuleFilePath, MBOrDesc.getError());

    StringRef ImageTarget =
        IsEmbeddedIR ? StringRef(EmbeddedIRTarget) : StringRef(RegularTarget);
    Images.emplace_back(std::move(*MBOrDesc), SI.Properties, SI.Symbols,
                        ImageTarget);
  }

  LLVMContext C;
  Module M("offload.wrapper.object", C);
  M.setTargetTriple(Triple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple())));

  auto CompileOptionsFromSYCLBackendCompileOptions =
      Args.getLastArgValue(OPT_sycl_backend_compile_options_EQ);
  auto LinkOptionsFromSYCLTargetLinkOptions =
      Args.getLastArgValue(OPT_sycl_target_link_options_EQ);

  StringRef CompileOptions(
      Args.MakeArgString(CompileOptionsFromSYCLBackendCompileOptions.str()));
  StringRef LinkOptions(
      Args.MakeArgString(LinkOptionsFromSYCLTargetLinkOptions.str()));
  offloading::SYCLWrappingOptions WrappingOptions;
  WrappingOptions.CompileOptions = CompileOptions;
  WrappingOptions.LinkOptions = LinkOptions;
  if (Verbose) {
    errs() << formatv(" offload-wrapper: compile-opts: {0}, link-opts: {1}\n",
                      CompileOptions, LinkOptions);
  }
  if (Error E = offloading::wrapSYCLBinaries(M, Images, WrappingOptions))
    return E;

  if (Args.hasArg(OPT_print_wrapped_module))
    errs() << "Wrapped Module\n" << M;

  // TODO: Once "clang tool->runCompile" migration is finished we need to remove
  // this scope and use community flow.
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(OutputFilePath, FD))
    return errorCodeToError(EC);

  raw_fd_ostream OS(FD, true);
  WriteBitcodeToFile(M, OS);
  return OutputFilePath;
}

/// Run clang tool for SYCL offloading.
/// 'InputFile' is the wrapped input file.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// clang tool.
static Expected<StringRef> runCompile(StringRef &InputFile,
                                      const ArgList &Args) {
  // Create a new file to write the output of clang to.
  auto OutputFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "o");
  if (!OutputFileOrErr)
    return OutputFileOrErr.takeError();

  Expected<std::string> ClangPath =
      findProgram("clang", {getMainExecutable("clang")});
  if (!ClangPath)
    return ClangPath.takeError();

  const llvm::Triple HostTriple(Args.getLastArgValue(OPT_host_triple_EQ));

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*ClangPath);

  const std::string TargetStr = "--target=" + HostTriple.getTriple();
  CmdArgs.push_back(TargetStr);

  // Checking for '-shared' linker option
  if (Args.hasArg(OPT_shared)) {
    if (!HostTriple.isOSWindows())
      CmdArgs.push_back("-fPIC");
  }
  CmdArgs.push_back("-c");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*OutputFileOrErr);
  CmdArgs.push_back(InputFile);
  if (Error Err = executeCommands(*ClangPath, CmdArgs))
    return std::move(Err);
  return *OutputFileOrErr;
}

/// Write an OffloadBinary containing the serialized SYCLBIN resulting from
/// \p ModuleDescs to the ExecutableName file with the .syclbin extension.
static Expected<StringRef>
packageSYCLBIN(SYCLBIN::BundleState State,
               const ArrayRef<SYCLBIN::SYCLBINModuleDesc> Modules) {
  SYCLBIN::SYCLBINDesc SYCLBIND{State, Modules};
  size_t SYCLBINByteSize = 0;
  if (Error E = SYCLBIND.getSYCLBINByteSite().moveInto(SYCLBINByteSize))
    return std::move(E);

  SmallString<0> SYCLBINImage;
  SYCLBINImage.reserve(SYCLBINByteSize);
  raw_svector_ostream SYCLBINImageOS{SYCLBINImage};
  if (Error E = SYCLBIN::write(SYCLBIND, SYCLBINImageOS))
    return std::move(E);

  OffloadingImage Image{};
  Image.TheImageKind = IMG_SYCLBIN;
  Image.TheOffloadKind = OFK_SYCL;
  Image.Image = MemoryBuffer::getMemBuffer(SYCLBINImage, /*BufferName=*/"",
                                           /*RequiresNullTerminator=*/false);

  std::unique_ptr<MemoryBuffer> Binary = MemoryBuffer::getMemBufferCopy(
      OffloadBinary::write(Image), Image.Image->getBufferIdentifier());

  auto OutFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "syclbin");
  if (!OutFileOrErr)
    return OutFileOrErr.takeError();

  Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
      FileOutputBuffer::create(*OutFileOrErr, Binary->getBufferSize());
  if (!OutputOrErr)
    return OutputOrErr.takeError();
  std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
  llvm::copy(Binary->getBuffer(), Output->getBufferStart());
  if (Error E = Output->commit())
    return std::move(E);

  return *OutFileOrErr;
}

Error copyFileToFinalExecutable(StringRef File, const ArgList &Args) {
  if (Verbose || DryRun) {
    llvm::Triple Triple(Args.getLastArgValue(OPT_host_triple_EQ,
                                             sys::getDefaultTargetTriple()));
    StringRef CopyCommand = Triple.isOSWindows() ? "copy" : "cp";
    llvm::errs() << "\"" << CopyCommand << "\" " << File << " "
                 << ExecutableName << "\n";
  }
  // TODO: check if copy can be replaced by rename.
  if (std::error_code EC = sys::fs::copy_file(File, ExecutableName))
    return createFileError(ExecutableName, EC);
  return Error::success();
}

Error mergeSYCLBIN(ArrayRef<StringRef> Files, const ArgList &Args) {
  // Fast path for the general case where there's only one file. In this case we
  // do not need to parse it and can instead simply copy it.
  if (Files.size() == 1) {
    if (Error Err = copyFileToFinalExecutable(Files[0], Args))
      reportError(std::move(Err));
    return Error::success();
  }
  // TODO: Merge SYCLBIN files here and write to ExecutableName output.
  // Use the first file as the base and modify.
  assert(Files.size() == 1);
  return Error::success();
}

// Run wrapping library and clang
static Expected<StringRef>
runWrapperAndCompile(std::vector<module_split::SplitModule> &SplitModules,
                     const ArgList &Args, bool IsEmbeddedIR = false) {
  auto OutputFile =
      sycl::wrapSYCLBinariesFromFile(SplitModules, Args, IsEmbeddedIR);
  if (!OutputFile)
    return OutputFile.takeError();
  // call to clang
  auto OutputFileOrErr = sycl::runCompile(*OutputFile, Args);
  if (!OutputFileOrErr)
    return OutputFileOrErr.takeError();
  return *OutputFileOrErr;
}

/// Link all SYCL device input files into one before adding device library
/// files. Device linking is performed using llvm-link tool.
/// 'InputFiles' is the list of all LLVM IR device input files.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// llvm-link tool.
Expected<StringRef> linkDeviceInputFiles(SmallVectorImpl<StringRef> &InputFiles,
                                         const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCL LinkDeviceInputFiles");

  Expected<std::string> LLVMLinkPath =
      findProgram("llvm-link", {getMainExecutable("llvm-link")});
  if (!LLVMLinkPath)
    return LLVMLinkPath.takeError();

  // Create a new file to write the linked device file to.
  auto OutFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "bc");
  if (!OutFileOrErr)
    return OutFileOrErr.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*LLVMLinkPath);
  for (auto &File : InputFiles) {
    auto IRFile = sycl::convertSPIRVToIR(File, Args);
    if (!IRFile)
      return IRFile.takeError();
    CmdArgs.push_back(*IRFile);
  }
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*OutFileOrErr);
  CmdArgs.push_back("--suppress-warnings");
  if (Error Err = executeCommands(*LLVMLinkPath, CmdArgs))
    return std::move(Err);
  return *OutFileOrErr;
}

/// Link all device library files and input file into one LLVM IR file. This
/// linking is performed using llvm-link tool.
/// 'InputFiles' is the list of all LLVM IR device input files.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// llvm-link tool.
static Expected<StringRef>
linkDeviceLibFiles(SmallVectorImpl<StringRef> &InputFiles,
                   const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("LinkDeviceLibraryFiles");

  Expected<std::string> LLVMLinkPath =
      findProgram("llvm-link", {getMainExecutable("llvm-link")});
  if (!LLVMLinkPath)
    return LLVMLinkPath.takeError();

  // Create a new file to write the linked device file to.
  auto OutFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "bc");
  if (!OutFileOrErr)
    return OutFileOrErr.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*LLVMLinkPath);
  CmdArgs.push_back("-only-needed");
  for (auto &File : InputFiles)
    CmdArgs.push_back(File);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*OutFileOrErr);
  CmdArgs.push_back("--suppress-warnings");
  if (Error Err = executeCommands(*LLVMLinkPath, CmdArgs))
    return std::move(Err);
  return *OutFileOrErr;
}

/// This function is used to link all SYCL device input files into a single
/// LLVM IR file. This file is in turn linked with all SYCL device library
/// files.
/// 'InputFiles' is the list of all LLVM IR device input files.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// llvm-link tool.
static Expected<StringRef> linkDevice(ArrayRef<StringRef> InputFiles,
                                      const ArgList &Args) {
  SmallVector<StringRef, 16> InputFilesVec;
  for (StringRef InputFile : InputFiles)
    InputFilesVec.emplace_back(InputFile);
  // First llvm-link step.
  auto LinkedFile = sycl::linkDeviceInputFiles(InputFilesVec, Args);
  if (!LinkedFile)
    reportError(LinkedFile.takeError());

  InputFilesVec.clear();
  InputFilesVec.emplace_back(*LinkedFile);

  // Gathering device library files
  SmallVector<std::string, 16> DeviceLibFiles;
  if (Error Err = sycl::getSYCLDeviceLibs(DeviceLibFiles, Args))
    reportError(std::move(Err));
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  SmallVector<std::string, 16> ExtractedDeviceLibFiles;
  for (auto &File : DeviceLibFiles) {
    auto BufferOrErr = MemoryBuffer::getFile(File);
    if (!BufferOrErr)
      return createFileError(File, BufferOrErr.getError());
    auto Buffer = std::move(*BufferOrErr);
    SmallVector<OffloadFile> Binaries;
    if (Error Err = extractOffloadBinaries(Buffer->getMemBufferRef(), Binaries))
      return std::move(Err);
    bool CompatibleBinaryFound = false;
    for (auto &Binary : Binaries) {
      auto BinTriple = Binary.getBinary()->getTriple();
      if (BinTriple == Triple.getTriple()) {
        auto FileNameOrErr =
            writeOffloadFile(Binary, true /* HasSYCLOffloadKind */);
        if (!FileNameOrErr)
          return FileNameOrErr.takeError();
        ExtractedDeviceLibFiles.emplace_back(*FileNameOrErr);
        CompatibleBinaryFound = true;
      }
    }
    if (!CompatibleBinaryFound)
      WithColor::warning(errs(), LinkerExecutable)
          << "Compatible SYCL device library binary not found\n";
  }

  // For NVPTX backend we need to also link libclc and CUDA libdevice.
  if (Triple.isNVPTX()) {
    if (Arg *A = Args.getLastArg(OPT_sycl_nvptx_device_lib_EQ)) {
      if (A->getValues().size() == 0)
        return createStringError(
            inconvertibleErrorCode(),
            "Number of device library files cannot be zero.");
      for (StringRef Val : A->getValues()) {
        SmallString<128> LibName(Val);
        if (llvm::sys::fs::exists(LibName))
          ExtractedDeviceLibFiles.emplace_back(std::string(LibName));
        else
          return createStringError(
              inconvertibleErrorCode(),
              std::string(LibName) +
                  " SYCL device library file for NVPTX is not found.");
      }
    }
  }

  // Make sure that SYCL device library files are available.
  // Note: For AMD targets, we do not pass any SYCL device libraries.
  if (ExtractedDeviceLibFiles.empty()) {
    // TODO: Add NVPTX when ready
    if (Triple.isSPIROrSPIRV())
      return createStringError(
          inconvertibleErrorCode(),
          " SYCL device library file list cannot be empty.");
    return *LinkedFile;
  }

  for (auto &File : ExtractedDeviceLibFiles)
    InputFilesVec.emplace_back(File);
  // second llvm-link step
  auto DeviceLinkedFile = sycl::linkDeviceLibFiles(InputFilesVec, Args);
  if (!DeviceLinkedFile)
    reportError(DeviceLinkedFile.takeError());

  return *DeviceLinkedFile;
}

static bool isStaticArchiveFile(const StringRef Filename) {
  if (!llvm::sys::path::has_extension(Filename))
    // Any file with no extension should not be considered an Archive.
    return false;
  llvm::file_magic Magic;
  llvm::identify_magic(Filename, Magic);
  // Only archive files are to be considered.
  // TODO: .lib check to be added
  return (Magic == llvm::file_magic::archive);
}

static Expected<StringRef> listSection(StringRef Filename,
                                       const ArgList &Args) {
  Expected<std::string> OffloadBundlerPath = findProgram(
      "clang-offload-bundler", {getMainExecutable("clang-offload-bundler")});
  if (!OffloadBundlerPath)
    return OffloadBundlerPath.takeError();
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*OffloadBundlerPath);
  bool IsArchive = isStaticArchiveFile(Filename);
  CmdArgs.push_back(IsArchive ? "-type=aoo" : "-type=o");
  CmdArgs.push_back(Saver.save("-input=" + Filename));
  CmdArgs.push_back("-list");
  auto Output = createOutputFile("bundled-targets", "list");
  if (!Output)
    return Output.takeError();
  SmallVector<std::optional<StringRef>> Redirects{std::nullopt, *Output,
                                                  std::nullopt};
  int ErrCode = llvm::sys::ExecuteAndWait(*OffloadBundlerPath, CmdArgs,
                                          std::nullopt, Redirects);
  if (ErrCode != 0)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to list targets");
  return *Output;
}

// This routine is used to run the clang-offload-bundler tool and unbundle
// device inputs that have been created with an older compiler where the
// device object is bundled into a host object.
static Expected<StringRef> unbundle(StringRef Filename, const ArgList &Args,
                                    llvm::Triple Triple) {
  Expected<std::string> OffloadBundlerPath = findProgram(
      "clang-offload-bundler", {getMainExecutable("clang-offload-bundler")});
  if (!OffloadBundlerPath)
    return OffloadBundlerPath.takeError();

  // Create a new file to write the unbundled file to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName), "ir");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*OffloadBundlerPath);
  bool IsArchive = isStaticArchiveFile(Filename);
  CmdArgs.push_back(IsArchive ? "-type=aoo" : "-type=o");
  auto *Target = Args.MakeArgString(Twine("-targets=sycl-") + Triple.str());
  CmdArgs.push_back(Target);
  CmdArgs.push_back(Saver.save("-input=" + Filename));
  CmdArgs.push_back(Saver.save("-output=" + *TempFileOrErr));
  CmdArgs.push_back("-unbundle");
  CmdArgs.push_back("-allow-missing-bundles");
  if (Error Err = executeCommands(*OffloadBundlerPath, CmdArgs))
    return std::move(Err);
  return *TempFileOrErr;
}

Error extractBundledObjects(StringRef Filename, const ArgList &Args,
                            SmallVector<OffloadFile> &Binaries) {
  auto List = listSection(Filename, Args);
  if (!List)
    return List.takeError();
  SmallVector<StringRef> TriplesInFile;
  llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> TripleList =
      llvm::MemoryBuffer::getFileOrSTDIN(*List, /*isText=*/true);
  if (std::error_code EC = TripleList.getError())
    return createFileError(*List, EC);
  (*TripleList)
      ->getBuffer()
      .split(TriplesInFile, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef TripleStr : TriplesInFile) {
    StringRef SYCLPrefix = "sycl-";
    if (!TripleStr.starts_with(SYCLPrefix))
      continue;
    llvm::Triple Triple(TripleStr.substr(SYCLPrefix.size()));
    auto UnbundledFile = unbundle(Filename, Args, Triple);
    if (!UnbundledFile)
      return UnbundledFile.takeError();
    if (*UnbundledFile == Filename)
      continue;

    SmallVector<StringRef> ObjectFilePaths;
    if (sycl::isStaticArchiveFile(Filename)) {
      llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> ObjList =
          llvm::MemoryBuffer::getFileOrSTDIN(*UnbundledFile, /*isText=*/true);
      if (std::error_code EC = ObjList.getError())
        return createFileError(*UnbundledFile, EC);
      // Create a copy of the list we can reference even after we close
      // the file.
      StringRef UnbundledArchiveList =
          Args.MakeArgString((*ObjList)->getBuffer());
      UnbundledArchiveList.split(ObjectFilePaths, '\n', /*MaxSplit=*/-1,
                                 /*KeepEmpty=*/false);
    } else {
      ObjectFilePaths.push_back(*UnbundledFile);
    }
    for (StringRef ObjectFilePath : ObjectFilePaths) {
      llvm::file_magic Magic;
      llvm::identify_magic(ObjectFilePath, Magic);
      if (Magic == file_magic::spirv_object)
        return createStringError(
            "SPIR-V fat objects must be generated with --offload-new-driver");
      const auto *Arg = Args.MakeArgString(
          "sycl-" +
          (Triple.isSPIROrSPIRV() ? Triple.str() + "-" : Triple.str()) + "=" +
          ObjectFilePath);
      auto Binary = getInputBitcodeLibrary(Arg);

      if (!Binary)
        return Binary.takeError();

      Binaries.push_back(std::move(*Binary));
    }
  }
  return Error::success();
}

} // namespace sycl

namespace generic {
Expected<StringRef> clang(ArrayRef<StringRef> InputFiles, const ArgList &Args,
                          bool IsSYCLKind = false) {
  llvm::TimeTraceScope TimeScope("Clang");
  // Use `clang` to invoke the appropriate device tools.
  Expected<std::string> ClangPath =
      findProgram("clang", {getMainExecutable("clang")});
  if (!ClangPath)
    return ClangPath.takeError();

  llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  llvm::Triple HostTriple(Args.getLastArgValue(OPT_host_triple_EQ));

  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);
  // Create a new file to write the linked device image to. Assume that the
  // input filename already has the device and architecture.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "." +
                           Triple.getArchName() + "." + Arch,
                       "img");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 16> CmdArgs{
      *ClangPath,
      "--no-default-config",
      "-o",
      *TempFileOrErr,
      Args.MakeArgString(
          "--target=" +
          (Triple.isNativeCPU() ? HostTriple : Triple).getTriple()),
  };

  if (!Arch.empty())
    Triple.isAMDGPU() ? CmdArgs.push_back(Args.MakeArgString("-mcpu=" + Arch))
                      : CmdArgs.push_back(Args.MakeArgString("-march=" + Arch));

  // AMDGPU is always in LTO mode currently.
  if (Triple.isAMDGPU())
    CmdArgs.push_back("-flto");

  // Forward all of the `--offload-opt` and similar options to the device.
  for (auto &Arg : Args.filtered(OPT_offload_opt_eq_minus, OPT_mllvm))
    CmdArgs.append(
        {"-Xlinker",
         Args.MakeArgString("--plugin-opt=" + StringRef(Arg->getValue()))});

  if (!Triple.isNVPTX() && !Triple.isSPIRV() && !Triple.isNativeCPU())
    CmdArgs.push_back("-Wl,--no-undefined");

  if (IsSYCLKind && Triple.isNVPTX())
    CmdArgs.push_back("-S");

  if (IsSYCLKind && Triple.isNativeCPU()) {
    CmdArgs.push_back("-Wno-override-module");
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back("-sycl-native-cpu-backend");
    CmdArgs.push_back("-c");
  }

  for (StringRef InputFile : InputFiles)
    CmdArgs.push_back(InputFile);

  // If this is CPU offloading we copy the input libraries.
  if (!Triple.isGPU() && !Triple.isNativeCPU()) {
    CmdArgs.push_back("-Wl,-Bsymbolic");
    CmdArgs.push_back("-shared");
    ArgStringList LinkerArgs;
    for (const opt::Arg *Arg :
         Args.filtered(OPT_INPUT, OPT_library, OPT_library_path, OPT_rpath,
                       OPT_whole_archive, OPT_no_whole_archive)) {
      // Sometimes needed libraries are passed by name, such as when using
      // sanitizers. We need to check the file magic for any libraries.
      if (Arg->getOption().matches(OPT_INPUT)) {
        if (!sys::fs::exists(Arg->getValue()) ||
            sys::fs::is_directory(Arg->getValue()))
          continue;

        file_magic Magic;
        if (auto EC = identify_magic(Arg->getValue(), Magic))
          return createStringError("Failed to open %s", Arg->getValue());
        if (Magic != file_magic::archive &&
            Magic != file_magic::elf_shared_object)
          continue;
      }
      if (Arg->getOption().matches(OPT_whole_archive))
        LinkerArgs.push_back(Args.MakeArgString("-Wl,--whole-archive"));
      else if (Arg->getOption().matches(OPT_no_whole_archive))
        LinkerArgs.push_back(Args.MakeArgString("-Wl,--no-whole-archive"));
      else
        Arg->render(Args, LinkerArgs);
    }
    llvm::append_range(CmdArgs, LinkerArgs);
  }

  // Pass on -mllvm options to the linker invocation.
  for (const opt::Arg *Arg : Args.filtered(OPT_mllvm))
    CmdArgs.append({"-Xlinker", Args.MakeArgString(
                                    "-mllvm=" + StringRef(Arg->getValue()))});

  if (SaveTemps && linkerSupportsLTO(Args))
    CmdArgs.push_back("-Wl,--save-temps");

  if (Args.hasArg(OPT_embed_bitcode))
    CmdArgs.push_back("-Wl,--lto-emit-llvm");

  for (StringRef Arg : Args.getAllArgValues(OPT_linker_arg_EQ))
    CmdArgs.append({"-Xlinker", Args.MakeArgString(Arg)});
  for (StringRef Arg : Args.getAllArgValues(OPT_compiler_arg_EQ))
    CmdArgs.push_back(Args.MakeArgString(Arg));

  for (StringRef Arg : Args.getAllArgValues(OPT_builtin_bitcode_EQ)) {
    if (llvm::Triple(Arg.split('=').first) == Triple)
      CmdArgs.append({"-Xclang", "-mlink-builtin-bitcode", "-Xclang",
                      Args.MakeArgString(Arg.split('=').second)});
  }

  // link NativeCPU utils lib if needed
  if (Triple.isNativeCPU()) {
    if (auto *A = Args.getLastArg(OPT_sycl_device_library_location_EQ)) {
      std::string NativeCPUUtilsLib = "";

      SmallVector<std::string, 8> LibraryPaths;
      for (const auto &Path : A->getValues()) {
        SmallString<128> LPath(Path);
        if (llvm::sys::fs::exists(LPath)) {
          LibraryPaths.emplace_back(LPath);
        }
      }

      for (auto &LPath : LibraryPaths) {
        // Call llvm-link without --only-needed to link to the nativecpu_utils
        // lib
        const char LibNativeCPUUtilsName[] = "libsycl-nativecpu_utils.bc";
        SmallString<128> LibNativeCPUUtilsPath(LPath);
        llvm::sys::path::append(LibNativeCPUUtilsPath, LibNativeCPUUtilsName);
        if (llvm::sys::fs::exists(LibNativeCPUUtilsPath)) {
          NativeCPUUtilsLib = LibNativeCPUUtilsPath.str();
          break;
        }
      }

      if (NativeCPUUtilsLib != "") {
        CmdArgs.append({"-Xclang", "-mlink-bitcode-file", "-Xclang",
                        Args.MakeArgString(NativeCPUUtilsLib)});
      }
    }
  }

  // The OpenMPOpt pass can introduce new calls and is expensive, we do
  // not want this when running CodeGen through clang.
  if (Args.hasArg(OPT_clang_backend) || Args.hasArg(OPT_builtin_bitcode_EQ))
    CmdArgs.append({"-mllvm", "-openmp-opt-disable"});

  if (Error Err = executeCommands(*ClangPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}
} // namespace generic

Expected<StringRef> linkDevice(ArrayRef<StringRef> InputFiles,
                               const ArgList &Args, bool IsSYCLKind = false) {
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  switch (Triple.getArch()) {
  case Triple::nvptx:
  case Triple::nvptx64:
  case Triple::amdgcn:
  case Triple::x86:
  case Triple::x86_64:
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::ppc64:
  case Triple::ppc64le:
  case Triple::systemz:
    return generic::clang(InputFiles, Args, IsSYCLKind);
  case Triple::spirv32:
  case Triple::spirv64:
  case Triple::spir:
  case Triple::spir64: {
    if (Triple.getSubArch() != llvm::Triple::NoSubArch &&
        Triple.getSubArch() != llvm::Triple::SPIRSubArch_gen &&
        Triple.getSubArch() != llvm::Triple::SPIRSubArch_x86_64)
      return createStringError(
          inconvertibleErrorCode(),
          "For SPIR targets, Linking is supported only for JIT compilations "
          "and AOT compilations for Intel CPUs/GPUs");
    if (IsSYCLKind) {
      auto SPVFile = sycl::runLLVMToSPIRVTranslation(InputFiles[0], Args);
      if (!SPVFile)
        return SPVFile.takeError();
      // TODO(NOM6): Add AOT support for other targets
      bool NeedAOTCompile =
          (Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen ||
           Triple.getSubArch() == llvm::Triple::SPIRSubArch_x86_64);
      auto AOTFile =
          (NeedAOTCompile) ? sycl::runAOTCompile(*SPVFile, Args) : *SPVFile;
      if (!AOTFile)
        return AOTFile.takeError();
      return NeedAOTCompile ? *AOTFile : *SPVFile;
    }
    // Return empty file
    return StringRef("");
  }
  case Triple::loongarch64:
    return generic::clang(InputFiles, Args, IsSYCLKind);
  case Triple::native_cpu:
    if (IsSYCLKind)
      return generic::clang(InputFiles, Args, IsSYCLKind);
    return createStringError(Triple.getArchName() +
                             " linking is not supported other than for SYCL");
  default:
    return createStringError(Triple.getArchName() +
                             " linking is not supported");
  }
}

// Compile the module to an object file using the appropriate target machine for
// the host triple.
Expected<StringRef> compileModule(Module &M, OffloadKind Kind) {
  llvm::TimeTraceScope TimeScope("Compile module");
  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return createStringError(Msg);

  auto Options =
      codegen::InitTargetOptionsFromCodeGenFlags(M.getTargetTriple());
  StringRef CPU = "";
  StringRef Features = "";
  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(M.getTargetTriple(), CPU, Features, Options,
                             Reloc::PIC_, M.getCodeModel()));

  if (M.getDataLayout().isDefault())
    M.setDataLayout(TM->createDataLayout());

  int FD = -1;
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "." +
                           getOffloadKindName(Kind) + ".image.wrapper",
                       "o");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();
  if (std::error_code EC = sys::fs::openFileForWrite(*TempFileOrErr, FD))
    return errorCodeToError(EC);

  auto OS = std::make_unique<llvm::raw_fd_ostream>(FD, true);

  legacy::PassManager CodeGenPasses;
  TargetLibraryInfoImpl TLII(M.getTargetTriple());
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));
  if (TM->addPassesToEmitFile(CodeGenPasses, *OS, nullptr,
                              CodeGenFileType::ObjectFile))
    return createStringError("Failed to execute host backend");
  CodeGenPasses.run(M);

  return *TempFileOrErr;
}

/// Creates the object file containing the device image and runtime
/// registration code from the device images stored in \p Images.
Expected<StringRef>
wrapDeviceImages(ArrayRef<std::unique_ptr<MemoryBuffer>> Buffers,
                 const ArgList &Args, OffloadKind Kind) {
  llvm::TimeTraceScope TimeScope("Wrap bundled images");

  SmallVector<ArrayRef<char>, 4> BuffersToWrap;
  for (const auto &Buffer : Buffers)
    BuffersToWrap.emplace_back(
        ArrayRef<char>(Buffer->getBufferStart(), Buffer->getBufferSize()));

  LLVMContext Context;
  Module M("offload.wrapper.module", Context);
  M.setTargetTriple(Triple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple())));

  switch (Kind) {
  case OFK_OpenMP:
    if (Error Err = offloading::wrapOpenMPBinaries(
            M, BuffersToWrap, offloading::getOffloadEntryArray(M),
            /*Suffix=*/"", /*Relocatable=*/Args.hasArg(OPT_relocatable)))
      return std::move(Err);
    break;
  case OFK_Cuda:
    if (Error Err = offloading::wrapCudaBinary(
            M, BuffersToWrap.front(), offloading::getOffloadEntryArray(M),
            /*Suffix=*/"", /*EmitSurfacesAndTextures=*/false))
      return std::move(Err);
    break;
  case OFK_HIP:
    if (Error Err = offloading::wrapHIPBinary(
            M, BuffersToWrap.front(), offloading::getOffloadEntryArray(M)))
      return std::move(Err);
    break;
  default:
    return createStringError(getOffloadKindName(Kind) +
                             " wrapping is not supported");
  }

  if (Args.hasArg(OPT_print_wrapped_module))
    errs() << M;
  if (Args.hasArg(OPT_save_temps)) {
    int FD = -1;
    auto TempFileOrErr =
        createOutputFile(sys::path::filename(ExecutableName) + "." +
                             getOffloadKindName(Kind) + ".image.wrapper",
                         "bc");
    if (!TempFileOrErr)
      return TempFileOrErr.takeError();
    if (std::error_code EC = sys::fs::openFileForWrite(*TempFileOrErr, FD))
      return errorCodeToError(EC);
    llvm::raw_fd_ostream OS(FD, true);
    WriteBitcodeToFile(M, OS);
  }

  auto FileOrErr = compileModule(M, Kind);
  if (!FileOrErr)
    return FileOrErr.takeError();
  return *FileOrErr;
}

Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleOpenMP(ArrayRef<OffloadingImage> Images) {
  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  for (const OffloadingImage &Image : Images)
    Buffers.emplace_back(
        MemoryBuffer::getMemBufferCopy(OffloadBinary::write(Image)));

  return std::move(Buffers);
}

Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleCuda(ArrayRef<OffloadingImage> Images, const ArgList &Args) {
  SmallVector<std::pair<StringRef, StringRef>, 4> InputFiles;
  for (const OffloadingImage &Image : Images)
    InputFiles.emplace_back(std::make_pair(Image.Image->getBufferIdentifier(),
                                           Image.StringData.lookup("arch")));

  auto FileOrErr = nvptx::fatbinary(InputFiles, Args);
  if (!FileOrErr)
    return FileOrErr.takeError();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ImageOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(*FileOrErr);

  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  if (std::error_code EC = ImageOrError.getError())
    return createFileError(*FileOrErr, EC);
  Buffers.emplace_back(std::move(*ImageOrError));

  return std::move(Buffers);
}

Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleHIP(ArrayRef<OffloadingImage> Images, const ArgList &Args) {
  SmallVector<std::pair<StringRef, StringRef>, 4> InputFiles;
  for (const OffloadingImage &Image : Images)
    InputFiles.emplace_back(std::make_pair(Image.Image->getBufferIdentifier(),
                                           Image.StringData.lookup("arch")));

  auto FileOrErr = amdgcn::fatbinary(InputFiles, Args);
  if (!FileOrErr)
    return FileOrErr.takeError();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ImageOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(*FileOrErr);

  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  if (std::error_code EC = ImageOrError.getError())
    return createFileError(*FileOrErr, EC);
  Buffers.emplace_back(std::move(*ImageOrError));

  return std::move(Buffers);
}

/// Transforms the input \p Images into the binary format the runtime expects
/// for the given \p Kind.
Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleLinkedOutput(ArrayRef<OffloadingImage> Images, const ArgList &Args,
                   OffloadKind Kind) {
  llvm::TimeTraceScope TimeScope("Bundle linked output");
  switch (Kind) {
  case OFK_OpenMP:
    return bundleOpenMP(Images);
  case OFK_Cuda:
    return bundleCuda(Images, Args);
  case OFK_HIP:
    return bundleHIP(Images, Args);
  default:
    return createStringError(getOffloadKindName(Kind) +
                             " bundling is not supported");
  }
}

/// Returns a new ArgList containg arguments used for the device linking phase.
DerivedArgList getLinkerArgs(ArrayRef<OffloadFile> Input,
                             const InputArgList &Args) {
  DerivedArgList DAL = DerivedArgList(DerivedArgList(Args));
  for (Arg *A : Args)
    DAL.append(A);

  // Set the subarchitecture and target triple for this compilation.
  const OptTable &Tbl = getOptTable();
  StringRef Arch = Args.MakeArgString(Input.front().getBinary()->getArch());
  DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_arch_EQ),
                   Arch == "generic" ? "" : Arch);
  DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_triple_EQ),
                   Args.MakeArgString(Input.front().getBinary()->getTriple()));

  const auto *Bin = Input.front().getBinary();
  DAL.AddJoinedArg(
      nullptr, Tbl.getOption(OPT_sycl_backend_compile_options_from_image_EQ),
      Args.MakeArgString(Bin->getString(COMPILE_OPTS)));
  DAL.AddJoinedArg(nullptr,
                   Tbl.getOption(OPT_sycl_backend_link_options_from_image_EQ),
                   Args.MakeArgString(Bin->getString(LINK_OPTS)));

  // If every input file is bitcode we have whole program visibility as we
  // do only support static linking with bitcode.
  auto ContainsBitcode = [](const OffloadFile &F) {
    return identify_magic(F.getBinary()->getImage()) == file_magic::bitcode;
  };
  if (llvm::all_of(Input, ContainsBitcode))
    DAL.AddFlagArg(nullptr, Tbl.getOption(OPT_whole_program));

  // Forward '-Xoffload-linker' options to the appropriate backend.
  for (StringRef Arg : Args.getAllArgValues(OPT_device_linker_args_EQ)) {
    auto [Triple, Value] = Arg.split('=');
    llvm::Triple TT(Triple);
    // If this isn't a recognized triple then it's an `arg=value` option.
    if (TT.getArch() == Triple::ArchType::UnknownArch)
      DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_linker_arg_EQ),
                       Args.MakeArgString(Arg));
    else if (Value.empty())
      DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_linker_arg_EQ),
                       Args.MakeArgString(Triple));
    else if (Triple == DAL.getLastArgValue(OPT_triple_EQ))
      DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_linker_arg_EQ),
                       Args.MakeArgString(Value));
  }

  // Forward '-Xoffload-compiler' options to the appropriate backend.
  for (StringRef Arg : Args.getAllArgValues(OPT_device_compiler_args_EQ)) {
    auto [Triple, Value] = Arg.split('=');
    llvm::Triple TT(Triple);
    // If this isn't a recognized triple then it's an `arg=value` option.
    if (TT.getArch() == Triple::ArchType::UnknownArch)
      DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_compiler_arg_EQ),
                       Args.MakeArgString(Arg));
    else if (Value.empty())
      DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_compiler_arg_EQ),
                       Args.MakeArgString(Triple));
    else if (Triple == DAL.getLastArgValue(OPT_triple_EQ))
      DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_compiler_arg_EQ),
                       Args.MakeArgString(Value));
  }

  return DAL;
}

Error handleOverrideImages(
    const InputArgList &Args,
    MapVector<OffloadKind, SmallVector<OffloadingImage, 0>> &Images) {
  for (StringRef Arg : Args.getAllArgValues(OPT_override_image)) {
    OffloadKind Kind = getOffloadKind(Arg.split("=").first);
    StringRef Filename = Arg.split("=").second;

    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(Filename);
    if (std::error_code EC = BufferOrErr.getError())
      return createFileError(Filename, EC);

    Expected<std::unique_ptr<ObjectFile>> ElfOrErr =
        ObjectFile::createELFObjectFile(**BufferOrErr,
                                        /*InitContent=*/false);
    if (!ElfOrErr)
      return ElfOrErr.takeError();
    ObjectFile &Elf = **ElfOrErr;

    OffloadingImage TheImage{};
    TheImage.TheImageKind = IMG_Object;
    TheImage.TheOffloadKind = Kind;
    TheImage.StringData["triple"] =
        Args.MakeArgString(Elf.makeTriple().getTriple());
    if (std::optional<StringRef> CPU = Elf.tryGetCPUName())
      TheImage.StringData["arch"] = Args.MakeArgString(*CPU);
    TheImage.Image = std::move(*BufferOrErr);

    Images[Kind].emplace_back(std::move(TheImage));
  }
  return Error::success();
}

/// Transforms all the extracted offloading input files into an image that can
/// be registered by the runtime.
Expected<SmallVector<StringRef>> linkAndWrapDeviceFiles(
    SmallVectorImpl<SmallVector<OffloadFile>> &LinkerInputFiles,
    const InputArgList &Args, char **Argv, int Argc) {
  llvm::TimeTraceScope TimeScope("Handle all device input");

  std::mutex ImageMtx;
  MapVector<OffloadKind, SmallVector<OffloadingImage, 0>> Images;
  // Create a binary image of each offloading image and embed it into a new
  // object file.
  SmallVector<StringRef> WrappedOutput;

  // When creating SYCLBIN files, we need to store the compiled modules for
  // combined packaging.
  std::mutex SYCLBINModulesMtx;
  SmallVector<SYCLBIN::SYCLBINModuleDesc> SYCLBINModules;

  // Initialize the images with any overriding inputs.
  if (Args.hasArg(OPT_override_image))
    if (Error Err = handleOverrideImages(Args, Images))
      return std::move(Err);

  auto Err = parallelForEachError(LinkerInputFiles, [&](auto &Input) -> Error {
    llvm::TimeTraceScope TimeScope("Link device input");

    // Each thread needs its own copy of the base arguments to maintain
    // per-device argument storage of synthetic strings.
    const OptTable &Tbl = getOptTable();
    BumpPtrAllocator Alloc;
    StringSaver Saver(Alloc);
    auto BaseArgs =
        Tbl.parseArgs(Argc, Argv, OPT_INVALID, Saver, [](StringRef Err) {
          reportError(createStringError(Err));
        });
    auto LinkerArgs = getLinkerArgs(Input, BaseArgs);
    bool HasSYCLOffloadKind = false;
    bool HasNonSYCLOffloadKinds = false;
    uint16_t ActiveOffloadKindMask = 0u;
    for (const auto &File : Input) {
      ActiveOffloadKindMask |= File.getBinary()->getOffloadKind();
      if (File.getBinary()->getOffloadKind() == OFK_SYCL)
        HasSYCLOffloadKind = true;
      else
        HasNonSYCLOffloadKinds = true;
    }

    // Write any remaining device inputs to an output file.
    SmallVector<StringRef> InputFiles;
    for (const OffloadFile &File : Input) {
      auto FileNameOrErr = writeOffloadFile(File);
      if (!FileNameOrErr)
        return FileNameOrErr.takeError();
      InputFiles.emplace_back(*FileNameOrErr);
    }

    if (HasSYCLOffloadKind) {
      SmallVector<StringRef> InputFiles;
      // Write device inputs to an output file for the linker.
      for (const OffloadFile &File : Input) {
        auto FileNameOrErr = writeOffloadFile(File, HasSYCLOffloadKind);
        if (!FileNameOrErr)
          return FileNameOrErr.takeError();
        InputFiles.emplace_back(*FileNameOrErr);
      }
      // Link the input device files using the device linker for SYCL
      // offload.
      auto TmpOutputOrErr = sycl::linkDevice(InputFiles, LinkerArgs);
      if (!TmpOutputOrErr)
        return TmpOutputOrErr.takeError();
      SmallVector<StringRef> InputFilesSYCL;
      InputFilesSYCL.emplace_back(*TmpOutputOrErr);
      auto SplitModulesOrErr =
          UseSYCLPostLinkTool
              ? sycl::runSYCLPostLinkTool(InputFilesSYCL, LinkerArgs)
              : sycl::runSYCLSplitLibrary(InputFilesSYCL, LinkerArgs,
                                          *SYCLModuleSplitMode);
      if (!SplitModulesOrErr)
        return SplitModulesOrErr.takeError();

      auto &SplitModules = *SplitModulesOrErr;
      const llvm::Triple Triple(LinkerArgs.getLastArgValue(OPT_triple_EQ));
      if ((Triple.isNVPTX() || Triple.isAMDGCN()) &&
          LinkerArgs.hasArg(OPT_sycl_embed_ir)) {
        // When compiling for Nvidia/AMD devices and the user requested the
        // IR to be embedded in the application (via option), run the output
        // of sycl-post-link (filetable referencing LLVM Bitcode + symbols)
        // through the offload wrapper and link the resulting object to the
        // application.
        auto OutputFile = sycl::runWrapperAndCompile(SplitModules, LinkerArgs,
                                                     /* IsEmbeddedIR */ true);
        if (!OutputFile)
          return OutputFile.takeError();
        WrappedOutput.push_back(*OutputFile);
      }
      for (size_t I = 0, E = SplitModules.size(); I != E; ++I) {
        SmallVector<StringRef> Files = {SplitModules[I].ModuleFilePath};
        StringRef Arch = LinkerArgs.getLastArgValue(OPT_arch_EQ);
        if (Arch.empty())
          Arch = "native";
        SmallVector<std::pair<StringRef, StringRef>, 4> BundlerInputFiles;
        auto ClangOutputOrErr =
            linkDevice(Files, LinkerArgs, true /* IsSYCLKind */);
        if (!ClangOutputOrErr)
          return ClangOutputOrErr.takeError();
        if (Triple.isNVPTX()) {
          auto VirtualArch = StringRef(clang::OffloadArchToVirtualArchString(
              clang::StringToOffloadArch(Arch)));
          auto PtxasOutputOrErr =
              nvptx::ptxas(*ClangOutputOrErr, LinkerArgs, Arch);
          if (!PtxasOutputOrErr)
            return PtxasOutputOrErr.takeError();
          BundlerInputFiles.emplace_back(*ClangOutputOrErr, VirtualArch);
          BundlerInputFiles.emplace_back(*PtxasOutputOrErr, Arch);
          auto BundledFileOrErr =
              nvptx::fatbinary(BundlerInputFiles, LinkerArgs);
          if (!BundledFileOrErr)
            return BundledFileOrErr.takeError();
          SplitModules[I].ModuleFilePath = *BundledFileOrErr;
        } else if (Triple.isAMDGCN()) {
          BundlerInputFiles.emplace_back(*ClangOutputOrErr, Arch);
          auto BundledFileOrErr =
              amdgcn::fatbinary(BundlerInputFiles, LinkerArgs);
          if (!BundledFileOrErr)
            return BundledFileOrErr.takeError();
          SplitModules[I].ModuleFilePath = *BundledFileOrErr;
        } else {
          SplitModules[I].ModuleFilePath = *ClangOutputOrErr;
          if (Triple.isNativeCPU()) {
            // Add to WrappedOutput directly rather than combining this with the
            // below because WrappedOutput holds references and
            // SplitModules[I].ModuleFilePath will go out of scope too soon.
            std::scoped_lock Guard(ImageMtx);
            WrappedOutput.push_back(*ClangOutputOrErr);
          }
        }
      }

      if (OutputSYCLBIN) {
        SYCLBIN::SYCLBINModuleDesc MD;
        MD.ArchString = LinkerArgs.getLastArgValue(OPT_arch_EQ);
        MD.TargetTriple =
            llvm::Triple{LinkerArgs.getLastArgValue(OPT_triple_EQ)};
        MD.SplitModules = std::move(SplitModules);
        std::scoped_lock<std::mutex> Guard(SYCLBINModulesMtx);
        SYCLBINModules.emplace_back(std::move(MD));
      } else {
        // TODO(NOM7): Remove this call and use community flow for bundle/wrap
        auto OutputFile = sycl::runWrapperAndCompile(SplitModules, LinkerArgs);
        if (!OutputFile)
          return OutputFile.takeError();

        // SYCL offload kind images are all ready to be sent to host linker.
        // TODO: Currently, device code wrapping for SYCL offload happens in a
        // separate path inside 'linkDevice' call seen above.
        // This will eventually be refactored to use the 'common' wrapping logic
        // that is used for other offload kinds.
        std::scoped_lock Guard(ImageMtx);
        WrappedOutput.push_back(*OutputFile);
      }
    }
    if (HasNonSYCLOffloadKinds) {
      // Write any remaining device inputs to an output file.
      SmallVector<StringRef> InputFiles;
      for (const OffloadFile &File : Input) {
        auto FileNameOrErr = writeOffloadFile(File);
        if (!FileNameOrErr)
          return FileNameOrErr.takeError();
        InputFiles.emplace_back(*FileNameOrErr);
      }

      // Link the remaining device files using the device linker.
      auto OutputOrErr = linkDevice(InputFiles, LinkerArgs);
      if (!OutputOrErr)
        return OutputOrErr.takeError();

      // Store the offloading image for each linked output file.
      for (OffloadKind Kind = OFK_OpenMP; Kind != OFK_LAST;
           Kind = static_cast<OffloadKind>((uint16_t)(Kind) << 1)) {
        if ((ActiveOffloadKindMask & Kind) == 0)
          continue;
        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(*OutputOrErr);
        if (std::error_code EC = FileOrErr.getError()) {
          if (DryRun)
            FileOrErr = MemoryBuffer::getMemBuffer("");
          else
            return createFileError(*OutputOrErr, EC);
        }

        // Manually containerize offloading images not in ELF format.
        if (Error E = containerizeRawImage(*FileOrErr, Kind, LinkerArgs))
          return E;

        std::scoped_lock<decltype(ImageMtx)> Guard(ImageMtx);
        OffloadingImage TheImage{};
        TheImage.TheImageKind =
            Args.hasArg(OPT_embed_bitcode) ? IMG_Bitcode : IMG_Object;
        TheImage.TheOffloadKind = Kind;
        TheImage.StringData["triple"] =
            Args.MakeArgString(LinkerArgs.getLastArgValue(OPT_triple_EQ));
        TheImage.StringData["arch"] =
            Args.MakeArgString(LinkerArgs.getLastArgValue(OPT_arch_EQ));
        TheImage.Image = std::move(*FileOrErr);

        Images[Kind].emplace_back(std::move(TheImage));
      }
    }
    return Error::success();
  });
  if (Err)
    return std::move(Err);

  if (OutputSYCLBIN) {
    auto OutputOrErr = sycl::packageSYCLBIN(SYCLBINState, SYCLBINModules);
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    WrappedOutput.push_back(*OutputOrErr);
  }

  for (auto &[Kind, Input] : Images) {
    if (Kind == OFK_SYCL)
      continue;
    // We sort the entries before bundling so they appear in a deterministic
    // order in the final binary.
    llvm::sort(Input, [](OffloadingImage &A, OffloadingImage &B) {
      return A.StringData["triple"] > B.StringData["triple"] ||
             A.StringData["arch"] > B.StringData["arch"] ||
             A.TheOffloadKind < B.TheOffloadKind;
    });
    auto BundledImagesOrErr = bundleLinkedOutput(Input, Args, Kind);
    if (!BundledImagesOrErr)
      return BundledImagesOrErr.takeError();
    auto OutputOrErr = wrapDeviceImages(*BundledImagesOrErr, Args, Kind);
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    WrappedOutput.push_back(*OutputOrErr);
  }
  return WrappedOutput;
}

std::optional<std::string> findFile(StringRef Dir, StringRef Root,
                                    const Twine &Name) {
  SmallString<128> Path;
  if (Dir.starts_with("="))
    sys::path::append(Path, Root, Dir.substr(1), Name);
  else
    sys::path::append(Path, Dir, Name);

  if (sys::fs::exists(Path))
    return static_cast<std::string>(Path);
  return std::nullopt;
}

std::optional<std::string>
findFromSearchPaths(StringRef Name, StringRef Root,
                    ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths)
    if (std::optional<std::string> File = findFile(Dir, Root, Name))
      return File;
  return std::nullopt;
}

std::optional<std::string>
searchLibraryBaseName(StringRef Name, StringRef Root,
                      ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths) {
    if (std::optional<std::string> File =
            findFile(Dir, Root, "lib" + Name + ".so"))
      return File;
    if (std::optional<std::string> File =
            findFile(Dir, Root, "lib" + Name + ".a"))
      return File;
  }
  return std::nullopt;
}

/// Search for static libraries in the linker's library path given input like
/// `-lfoo` or `-l:libfoo.a`.
std::optional<std::string> searchLibrary(StringRef Input, StringRef Root,
                                         ArrayRef<StringRef> SearchPaths) {
  if (Input.starts_with(":") || Input.ends_with(".lib"))
    return findFromSearchPaths(Input.drop_front(), Root, SearchPaths);
  return searchLibraryBaseName(Input, Root, SearchPaths);
}

/// Common redeclaration of needed symbol flags.
enum Symbol : uint32_t {
  Sym_None = 0,
  Sym_Undefined = 1U << 1,
  Sym_Weak = 1U << 2,
};

/// Scan the symbols from a BitcodeFile \p Buffer and record if we need to
/// extract any symbols from it.
Expected<bool> getSymbolsFromBitcode(MemoryBufferRef Buffer, OffloadKind Kind,
                                     bool IsArchive, StringSaver &Saver,
                                     DenseMap<StringRef, Symbol> &Syms) {
  Expected<IRSymtabFile> IRSymtabOrErr = readIRSymtab(Buffer);
  if (!IRSymtabOrErr)
    return IRSymtabOrErr.takeError();

  bool ShouldExtract = !IsArchive;
  DenseMap<StringRef, Symbol> TmpSyms;
  for (unsigned I = 0; I != IRSymtabOrErr->Mods.size(); ++I) {
    for (const auto &Sym : IRSymtabOrErr->TheReader.module_symbols(I)) {
      if (Sym.isFormatSpecific() || !Sym.isGlobal())
        continue;

      auto It = Syms.find(Sym.getName());
      bool NewSymbol = It == Syms.end();
      auto OldSym = NewSymbol ? Sym_None : It->second;

      // We will extract if it defines a currenlty undefined non-weak
      // symbol.
      bool ResolvesStrongReference =
          ((OldSym & Sym_Undefined && !(OldSym & Sym_Weak)) &&
           !Sym.isUndefined());
      // We will extract if it defines a new global symbol visible to the
      // host. This is only necessary for code targeting an offloading
      // language.
      bool NewGlobalSymbol =
          ((NewSymbol || (OldSym & Sym_Undefined)) && !Sym.isUndefined() &&
           !Sym.canBeOmittedFromSymbolTable() && Kind != object::OFK_None &&
           (Sym.getVisibility() != GlobalValue::HiddenVisibility));
      ShouldExtract |= ResolvesStrongReference | NewGlobalSymbol;

      // Update this symbol in the "table" with the new information.
      if (OldSym & Sym_Undefined && !Sym.isUndefined())
        TmpSyms[Saver.save(Sym.getName())] =
            static_cast<Symbol>(OldSym & ~Sym_Undefined);
      if (Sym.isUndefined() && NewSymbol)
        TmpSyms[Saver.save(Sym.getName())] =
            static_cast<Symbol>(OldSym | Sym_Undefined);
      if (Sym.isWeak())
        TmpSyms[Saver.save(Sym.getName())] =
            static_cast<Symbol>(OldSym | Sym_Weak);
    }
  }

  // If the file gets extracted we update the table with the new symbols.
  if (ShouldExtract)
    Syms.insert_range(TmpSyms);

  return ShouldExtract;
}

/// Scan the symbols from an ObjectFile \p Obj and record if we need to extract
/// any symbols from it.
Expected<bool> getSymbolsFromObject(const ObjectFile &Obj, OffloadKind Kind,
                                    bool IsArchive, StringSaver &Saver,
                                    DenseMap<StringRef, Symbol> &Syms) {
  bool ShouldExtract = !IsArchive;
  DenseMap<StringRef, Symbol> TmpSyms;
  for (SymbolRef Sym : Obj.symbols()) {
    auto FlagsOrErr = Sym.getFlags();
    if (!FlagsOrErr)
      return FlagsOrErr.takeError();

    if (!(*FlagsOrErr & SymbolRef::SF_Global) ||
        (*FlagsOrErr & SymbolRef::SF_FormatSpecific))
      continue;

    auto NameOrErr = Sym.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();

    bool NewSymbol = Syms.count(*NameOrErr) == 0;
    auto OldSym = NewSymbol ? Sym_None : Syms[*NameOrErr];

    // We will extract if it defines a currenlty undefined non-weak symbol.
    bool ResolvesStrongReference = (OldSym & Sym_Undefined) &&
                                   !(OldSym & Sym_Weak) &&
                                   !(*FlagsOrErr & SymbolRef::SF_Undefined);

    // We will extract if it defines a new global symbol visible to the
    // host. This is only necessary for code targeting an offloading
    // language.
    bool NewGlobalSymbol =
        ((NewSymbol || (OldSym & Sym_Undefined)) &&
         !(*FlagsOrErr & SymbolRef::SF_Undefined) && Kind != object::OFK_None &&
         !(*FlagsOrErr & SymbolRef::SF_Hidden));
    ShouldExtract |= ResolvesStrongReference | NewGlobalSymbol;

    // Update this symbol in the "table" with the new information.
    if (OldSym & Sym_Undefined && !(*FlagsOrErr & SymbolRef::SF_Undefined))
      TmpSyms[Saver.save(*NameOrErr)] =
          static_cast<Symbol>(OldSym & ~Sym_Undefined);
    if (*FlagsOrErr & SymbolRef::SF_Undefined && NewSymbol)
      TmpSyms[Saver.save(*NameOrErr)] =
          static_cast<Symbol>(OldSym | Sym_Undefined);
    if (*FlagsOrErr & SymbolRef::SF_Weak)
      TmpSyms[Saver.save(*NameOrErr)] = static_cast<Symbol>(OldSym | Sym_Weak);
  }

  // If the file gets extracted we update the table with the new symbols.
  if (ShouldExtract)
    Syms.insert_range(TmpSyms);

  return ShouldExtract;
}

/// Attempt to 'resolve' symbols found in input files. We use this to
/// determine if an archive member needs to be extracted. An archive member
/// will be extracted if any of the following is true.
///   1) It defines an undefined symbol in a regular object filie.
///   2) It defines a global symbol without hidden visibility that has not
///      yet been defined.
Expected<bool> getSymbols(StringRef Image, OffloadKind Kind, bool IsArchive,
                          StringSaver &Saver,
                          DenseMap<StringRef, Symbol> &Syms) {
  MemoryBufferRef Buffer = MemoryBufferRef(Image, "");
  switch (identify_magic(Image)) {
  case file_magic::bitcode:
    return getSymbolsFromBitcode(Buffer, Kind, IsArchive, Saver, Syms);
  case file_magic::elf_relocatable: {
    Expected<std::unique_ptr<ObjectFile>> ObjFile =
        ObjectFile::createObjectFile(Buffer);
    if (!ObjFile)
      return ObjFile.takeError();
    return getSymbolsFromObject(**ObjFile, Kind, IsArchive, Saver, Syms);
  }
  default:
    return false;
  }
}

/// Search the input files and libraries for embedded device offloading code
/// and add it to the list of files to be linked. Files coming from static
/// libraries are only added to the input if they are used by an existing
/// input file. Returns a list of input files intended for a single linking job.
Expected<SmallVector<SmallVector<OffloadFile>>>
getDeviceInput(const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("ExtractDeviceCode");

  // Skip all the input if the user is overriding the output.
  if (Args.hasArg(OPT_override_image))
    return SmallVector<SmallVector<OffloadFile>>();

  StringRef Root = Args.getLastArgValue(OPT_sysroot_EQ);
  SmallVector<StringRef> LibraryPaths;
  for (const opt::Arg *Arg : Args.filtered(OPT_library_path, OPT_libpath))
    LibraryPaths.push_back(Arg->getValue());

  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  // Try to extract device code from the linker input files.
  bool WholeArchive = Args.hasArg(OPT_wholearchive_flag) ? true : false;
  SmallVector<OffloadFile> ObjectFilesToExtract;
  SmallVector<OffloadFile> ArchiveFilesToExtract;
  for (const opt::Arg *Arg : Args.filtered(
           OPT_INPUT, OPT_library, OPT_whole_archive, OPT_no_whole_archive)) {
    if (Arg->getOption().matches(OPT_whole_archive) ||
        Arg->getOption().matches(OPT_no_whole_archive)) {
      WholeArchive = Arg->getOption().matches(OPT_whole_archive);
      continue;
    }

    std::optional<std::string> Filename =
        Arg->getOption().matches(OPT_library)
            ? searchLibrary(Arg->getValue(), Root, LibraryPaths)
            : std::string(Arg->getValue());

    if (!Filename && Arg->getOption().matches(OPT_library))
      reportError(
          createStringError("unable to find library -l%s", Arg->getValue()));

    if (!Filename || !sys::fs::exists(*Filename) ||
        sys::fs::is_directory(*Filename))
      continue;

    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFile(*Filename);
    if (std::error_code EC = BufferOrErr.getError())
      return createFileError(*Filename, EC);

    MemoryBufferRef Buffer = **BufferOrErr;
    if (identify_magic(Buffer.getBuffer()) == file_magic::elf_shared_object)
      continue;
    SmallVector<OffloadFile> Binaries;
    size_t OldSize = Binaries.size();
    if (Error Err = extractOffloadBinaries(Buffer, Binaries))
      return std::move(Err);
    if (Binaries.size() == OldSize) {
      if (Error Err = sycl::extractBundledObjects(*Filename, Args, Binaries))
        return std::move(Err);
    }

    for (auto &OffloadFile : Binaries) {
      if (identify_magic(Buffer.getBuffer()) == file_magic::archive &&
          !WholeArchive)
        ArchiveFilesToExtract.emplace_back(std::move(OffloadFile));
      else
        ObjectFilesToExtract.emplace_back(std::move(OffloadFile));
    }
  }

  // Link all standard input files and update the list of symbols.
  MapVector<OffloadFile::TargetID, SmallVector<OffloadFile, 0>> InputFiles;
  DenseMap<OffloadFile::TargetID, DenseMap<StringRef, Symbol>> Syms;
  for (OffloadFile &Binary : ObjectFilesToExtract) {
    if (!Binary.getBinary())
      continue;

    SmallVector<OffloadFile::TargetID> CompatibleTargets = {Binary};
    for (const auto &[ID, Input] : InputFiles)
      if (object::areTargetsCompatible(Binary, ID))
        CompatibleTargets.emplace_back(ID);

    for (const auto &[Index, ID] : llvm::enumerate(CompatibleTargets)) {
      Expected<bool> ExtractOrErr = getSymbols(
          Binary.getBinary()->getImage(), Binary.getBinary()->getOffloadKind(),
          /*IsArchive=*/false, Saver, Syms[ID]);
      if (!ExtractOrErr)
        return ExtractOrErr.takeError();

      // If another target needs this binary it must be copied instead.
      if (Index == CompatibleTargets.size() - 1)
        InputFiles[ID].emplace_back(std::move(Binary));
      else
        InputFiles[ID].emplace_back(Binary.copy());
    }
  }

  // Archive members only extract if they define needed symbols. We do this
  // after every regular input file so that libraries may be included out of
  // order. This follows 'ld.lld' semantics which are more lenient.
  bool Extracted = true;
  while (Extracted) {
    Extracted = false;
    for (OffloadFile &Binary : ArchiveFilesToExtract) {
      // If the binary was previously extracted it will be set to null.
      if (!Binary.getBinary())
        continue;

      SmallVector<OffloadFile::TargetID> CompatibleTargets = {Binary};
      for (const auto &[ID, Input] : InputFiles)
        if (object::areTargetsCompatible(Binary, ID))
          CompatibleTargets.emplace_back(ID);

      for (const auto &[Index, ID] : llvm::enumerate(CompatibleTargets)) {
        // Only extract an if we have an an object matching this target.
        if (!InputFiles.count(ID))
          continue;

        Expected<bool> ExtractOrErr =
            getSymbols(Binary.getBinary()->getImage(),
                       Binary.getBinary()->getOffloadKind(),
                       /*IsArchive=*/true, Saver, Syms[ID]);
        if (!ExtractOrErr)
          return ExtractOrErr.takeError();

        Extracted = *ExtractOrErr;

        // Skip including the file if it is an archive that does not resolve
        // any symbols.
        if (!Extracted)
          continue;

        // If another target needs this binary it must be copied instead.
        if (Index == CompatibleTargets.size() - 1)
          InputFiles[ID].emplace_back(std::move(Binary));
        else
          InputFiles[ID].emplace_back(Binary.copy());
      }

      // If we extracted any files we need to check all the symbols again.
      if (Extracted)
        break;
    }
  }

  for (StringRef Library : Args.getAllArgValues(OPT_bitcode_library_EQ)) {
    auto FileOrErr = getInputBitcodeLibrary(Library);
    if (!FileOrErr)
      return FileOrErr.takeError();
    InputFiles[*FileOrErr].push_back(std::move(*FileOrErr));
  }

  SmallVector<SmallVector<OffloadFile>> InputsForTarget;
  for (auto &[ID, Input] : InputFiles)
    InputsForTarget.emplace_back(std::move(Input));

  return std::move(InputsForTarget);
}

} // namespace

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  LinkerExecutable = Argv[0];
  sys::PrintStackTraceOnErrorSignal(Argv[0]);

  const OptTable &Tbl = getOptTable();
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  auto Args = Tbl.parseArgs(Argc, Argv, OPT_INVALID, Saver, [&](StringRef Err) {
    reportError(createStringError(Err));
  });

  if (Args.hasArg(OPT_help) || Args.hasArg(OPT_help_hidden)) {
    Tbl.printHelp(
        outs(),
        "clang-linker-wrapper [options] -- <options to passed to the linker>",
        "\nA wrapper utility over the host linker. It scans the input files\n"
        "for sections that require additional processing prior to linking.\n"
        "The will then transparently pass all arguments and input to the\n"
        "specified host linker to create the final binary.\n",
        Args.hasArg(OPT_help_hidden), Args.hasArg(OPT_help_hidden));
    return EXIT_SUCCESS;
  }
  if (Args.hasArg(OPT_v)) {
    printVersion(outs());
    return EXIT_SUCCESS;
  }

  // This forwards '-mllvm' arguments to LLVM if present.
  SmallVector<const char *> NewArgv = {Argv[0]};
  for (const opt::Arg *Arg : Args.filtered(OPT_mllvm))
    NewArgv.push_back(Arg->getValue());
  for (const opt::Arg *Arg : Args.filtered(OPT_offload_opt_eq_minus))
    NewArgv.push_back(Arg->getValue());
  SmallVector<PassPlugin, 1> PluginList;
  PassPlugins.setCallback([&](const std::string &PluginPath) {
    auto Plugin = PassPlugin::Load(PluginPath);
    if (!Plugin)
      reportFatalUsageError(Plugin.takeError());
    PluginList.emplace_back(Plugin.get());
  });
  cl::ParseCommandLineOptions(NewArgv.size(), &NewArgv[0]);

  Verbose = Args.hasArg(OPT_verbose);
  DryRun = Args.hasArg(OPT_dry_run);
  SaveTemps = Args.hasArg(OPT_save_temps);
  CudaBinaryPath = Args.getLastArgValue(OPT_cuda_path_EQ).str();

  llvm::Triple Triple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple()));
  if (Args.hasArg(OPT_o))
    ExecutableName = Args.getLastArgValue(OPT_o, "a.out");
  else if (Args.hasArg(OPT_out))
    ExecutableName = Args.getLastArgValue(OPT_out, "a.exe");
  else
    ExecutableName = Triple.isOSWindows() ? "a.exe" : "a.out";

  parallel::strategy = hardware_concurrency(1);
  if (auto *Arg = Args.getLastArg(OPT_wrapper_jobs)) {
    unsigned Threads = 0;
    if (!llvm::to_integer(Arg->getValue(), Threads) || Threads == 0)
      reportError(createStringError("%s: expected a positive integer, got '%s'",
                                    Arg->getSpelling().data(),
                                    Arg->getValue()));
    parallel::strategy = hardware_concurrency(Threads);
  }

  if (Args.hasArg(OPT_wrapper_time_trace_eq)) {
    unsigned Granularity;
    Args.getLastArgValue(OPT_wrapper_time_trace_granularity, "500")
        .getAsInteger(10, Granularity);
    timeTraceProfilerInitialize(Granularity, Argv[0]);
  }

  UseSYCLPostLinkTool = Args.hasFlag(OPT_use_sycl_post_link_tool,
                                     OPT_no_use_sycl_post_link_tool, true);
  if (!UseSYCLPostLinkTool && Args.hasArg(OPT_use_sycl_post_link_tool))
    reportError(createStringError("-use-sycl-post-link-tool and "
                                  "-no-use-sycl-post-link-tool options can't "
                                  "be used together."));

  if (Arg *SYCLBINArg = Args.getLastArg(OPT_syclbin_EQ)) {
    OutputSYCLBIN = true;
    auto MaybeSYCLBINState =
        llvm::StringSwitch<std::optional<SYCLBIN::BundleState>>(
            SYCLBINArg->getValue())
            .Case("input", SYCLBIN::BundleState::Input)
            .Case("object", SYCLBIN::BundleState::Object)
            .Case("executable", SYCLBIN::BundleState::Executable)
            .Default(std::nullopt);

    if (MaybeSYCLBINState)
      SYCLBINState = *MaybeSYCLBINState;
    else
      reportError(
          createStringError("Invalid argument for -syclbin (" +
                            StringRef{SYCLBINArg->getValue()} +
                            "). Must be either input, object or executable."));
  }

  if (OutputSYCLBIN && Args.hasArg(OPT_sycl_embed_ir))
    reportError(createStringError(
        "-sycl-embed_ir and -syclbin can't be used together."));

  if (Args.hasArg(OPT_sycl_module_split_mode_EQ)) {
    if (UseSYCLPostLinkTool)
      reportError(createStringError(
          "-sycl-module-split-mode should be used with "
          "the -no-use-sycl-post-link-tool command line option."));

    StringRef StrMode = Args.getLastArgValue(OPT_sycl_module_split_mode_EQ);
    SYCLModuleSplitMode = module_split::convertStringToSplitMode(StrMode);
    if (!SYCLModuleSplitMode)
      reportError(createStringError(
          inconvertibleErrorCode(),
          formatv("sycl-module-split-mode value isn't recognized: {0}",
                  StrMode)));
  }

  if (Args.hasArg(OPT_sycl_dump_device_code_EQ)) {
    Arg *A = Args.getLastArg(OPT_sycl_dump_device_code_EQ);
    OffloadImageDumpDir = A->getValue();
    if (OffloadImageDumpDir.empty())
      sys::path::native(OffloadImageDumpDir = "./");
    else
      OffloadImageDumpDir.append(sys::path::get_separator());
  }

  {
    llvm::TimeTraceScope TimeScope("Execute linker wrapper");

    // Extract the device input files stored in the host fat binary.
    auto DeviceInputFiles = getDeviceInput(Args);
    if (!DeviceInputFiles)
      reportError(DeviceInputFiles.takeError());

    // Link and wrap the device images extracted from the linker input.
    auto FilesOrErr =
        linkAndWrapDeviceFiles(*DeviceInputFiles, Args, Argv, Argc);
    if (!FilesOrErr)
      reportError(FilesOrErr.takeError());

    if (OutputSYCLBIN) {
      if (Error Err = sycl::mergeSYCLBIN(*FilesOrErr, Args))
        reportError(std::move(Err));
    } else if (Args.hasArg(OPT_sycl_device_link)) {
      // Skip host linker if --sycl-device-link option is set.
      // Just copy the output of device linking and wrapping action.
      if (FilesOrErr->size() != 1)
        reportError(
            createStringError("Expect single output from the device linker."));
      if (Error Err = sycl::copyFileToFinalExecutable((*FilesOrErr)[0], Args))
        reportError(std::move(Err));
    } else {
      // Run the host linking job with the rendered arguments.
      if (Error Err = runLinker(*FilesOrErr, Args))
        reportError(std::move(Err));
    }
  }

  if (const opt::Arg *Arg = Args.getLastArg(OPT_wrapper_time_trace_eq)) {
    if (Error Err = timeTraceProfilerWrite(Arg->getValue(), ExecutableName))
      reportError(std::move(Err));
    timeTraceProfilerCleanup();
  }

  // Remove the temporary files created.
  if (!SaveTemps)
    for (const auto &TempFile : TempFiles)
      if (std::error_code EC = sys::fs::remove(TempFile))
        reportError(createFileError(TempFile, EC));
  return EXIT_SUCCESS;
}
