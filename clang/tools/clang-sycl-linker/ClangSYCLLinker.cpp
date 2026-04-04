//=-------- clang-sycl-linker/ClangSYCLLinker.cpp - SYCL Linker util -------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This tool executes a sequence of steps required to link device code in SYCL
// device images. SYCL device code linking requires a complex sequence of steps
// that include linking of llvm bitcode files, linking device library files
// with the fully linked source bitcode file(s), running several SYCL specific
// post-link steps on the fully linked bitcode file(s), and finally generating
// target-specific device code.
//
//===---------------------------------------------------------------------===//

#include "clang/Basic/OffloadArch.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/TargetID.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/WithColor.h"
#include "llvm/SYCLPostLink/SYCLPostLink.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <mutex>

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::object;
using namespace clang;

/// Save intermediary results.
static bool SaveTemps = false;

/// Print commands/steps with arguments without executing.
static bool DryRun = false;

/// Print verbose output.
static bool Verbose = false;

/// Filename of the output being created.
static StringRef OutputFile;

/// Directory to dump SPIR-V IR if requested by user.
static SmallString<128> SPIRVDumpDir;

/// Mutex lock to protect writes to shared TempFiles in parallel.
static std::mutex TempFilesMutex;

using OffloadingImage = OffloadBinary::OffloadingImage;

static void printVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-sycl-linker") << '\n';
}

/// The value of `argv[0]` when run.
static const char *Executable;

/// Temporary files to be cleaned up.
static SmallVector<SmallString<128>> TempFiles;

namespace {
// Must not overlap with llvm::opt::DriverFlag.
enum LinkerFlags { LinkerOnlyOption = (1 << 4) };

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "SYCLLinkOpts.inc"
  LastOption
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "SYCLLinkOpts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "SYCLLinkOpts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

static constexpr OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "SYCLLinkOpts.inc"
#undef OPTION
};

class LinkerOptTable : public opt::GenericOptTable {
public:
  LinkerOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {}
};

const OptTable &getOptTable() {
  static const LinkerOptTable *Table = []() {
    auto Result = std::make_unique<LinkerOptTable>();
    return Result.release();
  }();
  return *Table;
}

[[noreturn]] void reportError(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), Executable));
  exit(EXIT_FAILURE);
}

std::string getMainExecutable(const char *Name) {
  void *Ptr = (void *)(intptr_t)&getMainExecutable;
  auto COWPath = sys::fs::getMainExecutable(Name, Ptr);
  return sys::path::parent_path(COWPath).str();
}

Expected<StringRef> createTempFile(const ArgList &Args, const Twine &Prefix,
                                   StringRef Extension) {
  SmallString<128> TempOutput;
  if (Args.hasArg(OPT_save_temps)) {
    sys::fs::createUniquePath(Prefix + "-%%%%%%." + Extension, TempOutput,
                              /*MakeAbsolute=*/false);
  } else {
    if (std::error_code EC =
            sys::fs::createTemporaryFile(Prefix, Extension, TempOutput))
      return createFileError(TempOutput, EC);
  }

  TempFiles.emplace_back(std::move(TempOutput));
  return TempFiles.back();
}

Expected<std::string> findProgram(const ArgList &Args, StringRef Name,
                                  ArrayRef<StringRef> Paths) {
  if (Args.hasArg(OPT_dry_run))
    return Name.str();
  ErrorOr<std::string> Path = sys::findProgramByName(Name, Paths);
  if (!Path)
    Path = sys::findProgramByName(Name);
  if (!Path)
    return createStringError(Path.getError(),
                             "Unable to find '" + Name + "' in path");
  return *Path;
}

void printCommands(ArrayRef<StringRef> CmdArgs) {
  if (CmdArgs.empty())
    return;

  llvm::errs() << " \"" << CmdArgs.front() << "\" ";
  llvm::errs() << llvm::join(std::next(CmdArgs.begin()), CmdArgs.end(), " ")
               << "\n";
}

namespace sycl {
  // This utility function is used to gather all SYCL device library files that
  // will be linked with input device files.
  // The list of files and its location are passed from driver.
  static Error getSYCLDeviceLibs(SmallVector<std::string, 16> &DeviceLibFiles,
                                const ArgList &Args) {
    StringRef SYCLDeviceLibLoc("");
    if (Arg *A = Args.getLastArg(OPT_library_path_EQ))
      SYCLDeviceLibLoc = A->getValue();
    if (Arg *A = Args.getLastArg(OPT_device_libs_EQ)) {
      llvm::errs() << "[DEBUG] SYCL device library location: " << SYCLDeviceLibLoc << "\n";
      llvm::errs() << "[DEBUG] SYCL device libraries to link: " << llvm::join(A->getValues(), ", ") << "\n";
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
} // namespace sycl


/// Get a temporary filename suitable for output.
Expected<StringRef> createOutputFile(const Twine &Prefix, StringRef Extension) {
  std::scoped_lock<decltype(TempFilesMutex)> Lock(TempFilesMutex);
  SmallString<128> OutputFile;
  std::string PrefixStr = clang::sanitizeTargetIDInFileName(Prefix.str());

  if (SaveTemps) {
    // Generate a unique path name without creating a file
    sys::fs::createUniquePath(Prefix + "-%%%%%%." + Extension, OutputFile,
                              /*MakeAbsolute=*/false);
    (PrefixStr + "." + Extension).toNullTerminatedStringRef(OutputFile);
  } else {
    if (std::error_code EC =
            sys::fs::createTemporaryFile(PrefixStr, Extension, OutputFile))
      return createFileError(OutputFile, EC);
  }

  TempFiles.emplace_back(std::move(OutputFile));
  return TempFiles.back();
}

// TODO: Remove HasSYCLOffloadKind dependence when aligning with community code.
Expected<StringRef> writeOffloadFile(const OffloadFile &File,
                                     bool HasSYCLOffloadKind = false) {
  const OffloadBinary &Binary = *File.getBinary();

  StringRef Prefix =
      sys::path::stem(Binary.getMemoryBufferRef().getBufferIdentifier());

  StringRef BinArch = (Binary.getArch() == "*") ? "any" : Binary.getArch();
  auto TempFileOrErr = createOutputFile(
      Prefix + "-" + Binary.getTriple() + "-" + BinArch,
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

Expected<SmallVector<std::string>> getInput(const ArgList &Args) {
  // Collect all input bitcode files to be passed to the device linking stage.
  SmallVector<std::string> BitcodeFiles;
  for (const opt::Arg *Arg : Args.filtered(OPT_input_file_EQ)) {
    std::optional<std::string> Filename = std::string(Arg->getValue());
    if (!Filename || !sys::fs::exists(*Filename) ||
        sys::fs::is_directory(*Filename))
      continue;
    file_magic Magic;
    if (auto EC = identify_magic(*Filename, Magic))
      return createStringError("Failed to open file " + *Filename);
    if (Magic == file_magic::elf_shared_object ||
        Magic == file_magic::elf_relocatable  ||
        Magic == file_magic::elf_executable   ||
        Magic == file_magic::spirv_object     ||
        Magic == file_magic::archive)
      continue;
    if (Magic != file_magic::bitcode)
      return createStringError("Unsupported file type");
    BitcodeFiles.push_back(*Filename);
  }
  return BitcodeFiles;
}

/// Handle cases where input file is a LLVM IR bitcode file.
Expected<std::unique_ptr<Module>> getBitcodeModule(StringRef File,
                                                   LLVMContext &C) {
  SMDiagnostic Err;
  auto M = getLazyIRFileModule(File, Err, C);
  if (M)
    return std::move(M);
  return createStringError(Err.getMessage());
}

/// Link all SYCL device input files into one using llvm-link.
static Expected<StringRef>
linkDeviceInputFiles(ArrayRef<std::string> InputFiles, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCL LinkDeviceInputFiles");

  Expected<std::string> Linker =
      findProgram(Args, "llvm-link", {getMainExecutable("llvm-link")});
  if (!Linker)
    return Linker.takeError();

  auto OutFileOrErr =
      createTempFile(Args, sys::path::filename(OutputFile), "bc");
  if (!OutFileOrErr)
    return OutFileOrErr.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*Linker);
  CmdArgs.push_back("--suppress-warnings");
  for (auto &File : InputFiles)
    CmdArgs.push_back(File);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*OutFileOrErr);

  if (Error Err = executeCommands(*Linker, CmdArgs))
    return std::move(Err);
  return *OutFileOrErr;
}

/// Link device library files with the linked input using llvm-link -only-needed.
static Expected<StringRef>
linkDeviceLibFiles(SmallVectorImpl<StringRef> &InputFiles,
                   const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("LinkDeviceLibraryFiles");

  Expected<std::string> Linker =
      findProgram(Args, "llvm-link", {getMainExecutable("llvm-link")});
  if (!Linker)
    return Linker.takeError();

  auto OutFileOrErr =
      createTempFile(Args, sys::path::filename(OutputFile), "bc");
  if (!OutFileOrErr)
    return OutFileOrErr.takeError();

  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*Linker);
  CmdArgs.push_back("-only-needed");
  CmdArgs.push_back("--suppress-warnings");
  for (auto &File : InputFiles)
    CmdArgs.push_back(File);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*OutFileOrErr);

  if (Error Err = executeCommands(*Linker, CmdArgs))
    return std::move(Err);
  return *OutFileOrErr;
}

/// Following tasks are performed:
/// 1. Link all SYCL device bitcode images into one image using llvm-link.
/// 2. Gather all SYCL device library bitcode images, extracting from offload
///    binaries and matching by triple.
/// 3. Link Step 2 results with Step 1 output using llvm-link -only-needed.
static Expected<StringRef> linkDeviceCode(ArrayRef<std::string> InputFiles,
                                          const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCL link device code");
  assert(InputFiles.size() && "No inputs to link");

  // Step 1: link all input bitcode files together.
  auto LinkedFile = linkDeviceInputFiles(InputFiles, Args);
  if (!LinkedFile)
    return LinkedFile.takeError();

  // Step 2: gather device library files and extract bitcode matching by triple.
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  
  // Get SYCL device libraries
  SmallVector<std::string, 16> DeviceLibFiles;
  if (Error Err = sycl::getSYCLDeviceLibs(DeviceLibFiles, Args))
    return std::move(Err);
  SmallVector<std::string, 16> ExtractedDeviceLibFiles;
  for (auto &File : DeviceLibFiles) {
    auto BufferOrErr = MemoryBuffer::getFile(File);
    if (!BufferOrErr)
      return createFileError(File, BufferOrErr.getError());
    auto Buffer = std::move(*BufferOrErr);
    SmallVector<OffloadFile, 16> Binaries;
    if (Error Err = extractOffloadBinaries(Buffer->getMemBufferRef(), Binaries))
      return std::move(Err);
    bool CompatibleBinaryFound = false;
    for (auto &Binary : Binaries) {
      auto BinTriple = Binary.getBinary()->getTriple();
      if (BinTriple == Triple.getTriple()) {
        auto FileNameOrErr = writeOffloadFile(Binary, true /* HasSYCLOffloadKind */);
        if (!FileNameOrErr)
          return FileNameOrErr.takeError();
        ExtractedDeviceLibFiles.emplace_back(*FileNameOrErr);
        CompatibleBinaryFound = true;
      }
    }
    if (!CompatibleBinaryFound)
      WithColor::warning(errs(), Executable)
          << "Compatible SYCL device library binary not found for: " << File << "\n";
  }

  // Handle explicit bitcode libraries from command line
  for (StringRef Library : Args.getAllArgValues(OPT_bitcode_library_EQ)) {
    auto [LibraryTriple, LibraryPath] = Library.split('=');
    if (llvm::Triple(LibraryTriple) != Triple)
      continue;

    if (!llvm::sys::fs::exists(LibraryPath))
      return createStringError(inconvertibleErrorCode(),
                               "The specified device library " + LibraryPath +
                                   " does not exist.");

    ExtractedDeviceLibFiles.emplace_back(LibraryPath.str());
  }

  // If no device libs found, return step 1 output directly.
  if (ExtractedDeviceLibFiles.empty()) {
    if (Triple.isSPIROrSPIRV())
      WithColor::warning(errs(), Executable)
          << "SYCL device library file list is empty\n";
    return *LinkedFile;
  }

  // Step 3: link device libs with -only-needed.
  SmallVector<StringRef, 16> InputFilesVec;
  InputFilesVec.push_back(*LinkedFile);
  for (auto &File : ExtractedDeviceLibFiles)
    InputFilesVec.push_back(File);

  auto DeviceLinkedFile = linkDeviceLibFiles(InputFilesVec, Args);
  if (!DeviceLinkedFile)
    return DeviceLinkedFile.takeError();

  return *DeviceLinkedFile;
}

/// Run AOT compilation for Intel CPU.
static Error runAOTCompileIntelCPU(StringRef InputFile, StringRef OutputFile,
                                   const ArgList &Args) {
  SmallVector<StringRef, 8> CmdArgs;
  Expected<std::string> OpenCLAOTPath =
      findProgram(Args, "opencl-aot", {getMainExecutable("opencl-aot")});
  if (!OpenCLAOTPath)
    return OpenCLAOTPath.takeError();

  CmdArgs.push_back(*OpenCLAOTPath);
  CmdArgs.push_back("--device=cpu");
  StringRef ExtraArgs = Args.getLastArgValue(OPT_opencl_aot_options_EQ);
  ExtraArgs.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(OutputFile);
  CmdArgs.push_back(InputFile);
  if (Error Err = executeCommands(*OpenCLAOTPath, CmdArgs))
    return Err;
  return Error::success();
}

/// Run AOT compilation for Intel GPU.
static Error runAOTCompileIntelGPU(StringRef InputFile, StringRef OutputFile,
                                   const ArgList &Args) {
  SmallVector<StringRef, 8> CmdArgs;
  Expected<std::string> OclocPath =
      findProgram(Args, "ocloc", {getMainExecutable("ocloc")});
  if (!OclocPath)
    return OclocPath.takeError();

  CmdArgs.push_back(*OclocPath);
  CmdArgs.push_back("-output_no_suffix");
  CmdArgs.push_back("-spirv_input");

  StringRef Arch(Args.getLastArgValue(OPT_arch_EQ));
  if (Arch.empty())
    return createStringError(inconvertibleErrorCode(),
                             "Arch must be specified for AOT compilation");
  CmdArgs.push_back("-device");
  CmdArgs.push_back(Arch);

  StringRef ExtraArgs = Args.getLastArgValue(OPT_ocloc_options_EQ);
  ExtraArgs.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  CmdArgs.push_back("-output");
  CmdArgs.push_back(OutputFile);
  CmdArgs.push_back("-file");
  CmdArgs.push_back(InputFile);
  if (Error Err = executeCommands(*OclocPath, CmdArgs))
    return Err;
  return Error::success();
}

/// Run AOT compilation for Intel CPU/GPU.
static Error runAOTCompile(StringRef InputFile, StringRef OutputFile,
                           const ArgList &Args) {
  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);
  OffloadArch OA = StringToOffloadArch(Arch);
  if (IsIntelGPUOffloadArch(OA))
    return runAOTCompileIntelGPU(InputFile, OutputFile, Args);
  if (IsIntelCPUOffloadArch(OA))
    return runAOTCompileIntelCPU(InputFile, OutputFile, Args);
  return createStringError(inconvertibleErrorCode(), "Unsupported arch");
}

// TODO: Consider using LLVM-IR metadata to identify globals of interest
bool isKernel(const Function &F) {
  const llvm::CallingConv::ID CC = F.getCallingConv();
  return CC == llvm::CallingConv::SPIR_KERNEL ||
         CC == llvm::CallingConv::AMDGPU_KERNEL ||
         CC == llvm::CallingConv::PTX_Kernel;
}

/// Add any llvm-spirv option that relies on a specific Triple in addition
/// to user supplied options.
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

  std::string ExtArg("-spirv-ext=-all");
  ExtArg +=
      ",+SPV_EXT_shader_atomic_float_add"
      ",+SPV_EXT_shader_atomic_float_min_max"
      ",+SPV_KHR_no_integer_wrap_decoration"
      ",+SPV_KHR_float_controls"
      ",+SPV_KHR_expect_assume"
      ",+SPV_KHR_linkonce_odr"
      ",+SPV_INTEL_subgroups"
      ",+SPV_INTEL_media_block_io"
      ",+SPV_INTEL_device_side_avc_motion_estimation"
      ",+SPV_INTEL_fpga_loop_controls"
      ",+SPV_INTEL_unstructured_loop_controls"
      ",+SPV_INTEL_fpga_reg"
      ",+SPV_INTEL_blocking_pipes"
      ",+SPV_INTEL_function_pointers"
      ",+SPV_INTEL_kernel_attributes"
      ",+SPV_INTEL_io_pipes"
      ",+SPV_INTEL_inline_assembly"
      ",+SPV_INTEL_arbitrary_precision_integers"
      ",+SPV_INTEL_float_controls2"
      ",+SPV_INTEL_vector_compute"
      ",+SPV_INTEL_arbitrary_precision_fixed_point"
      ",+SPV_INTEL_arbitrary_precision_floating_point"
      ",+SPV_INTEL_variable_length_array"
      ",+SPV_INTEL_fp_fast_math_mode"
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
      ",+SPV_INTEL_task_sequence"
      ",+SPV_INTEL_bfloat16_conversion"
      ",+SPV_INTEL_joint_matrix"
      ",+SPV_INTEL_hw_thread_queries"
      ",+SPV_KHR_uniform_group_instructions"
      ",+SPV_INTEL_masked_gather_scatter"
      ",+SPV_INTEL_tensor_float32_conversion"
      ",+SPV_INTEL_optnone"
      ",+SPV_KHR_non_semantic_info"
      ",+SPV_KHR_cooperative_matrix"
      ",+SPV_EXT_shader_atomic_float16_add"
      ",+SPV_INTEL_fp_max_error"
      ",+SPV_INTEL_memory_access_aliasing";
  TranslatorArgs.push_back(Args.MakeArgString(ExtArg));
}

/// Run LLVM to SPIR-V translation.
static Expected<StringRef> runLLVMToSPIRVTranslation(StringRef File,
                                                      const ArgList &Args) {
  Expected<std::string> LLVMToSPIRVPath =
      findProgram(Args, "llvm-spirv", {getMainExecutable("llvm-spirv")});
  if (!LLVMToSPIRVPath)
    return LLVMToSPIRVPath.takeError();

  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*LLVMToSPIRVPath);

  getTripleBasedSPIRVTransOpts(Args, CmdArgs, Triple);

  StringRef LLVMToSPIRVOptions = Args.getLastArgValue(OPT_llvm_spirv_options_EQ);
  LLVMToSPIRVOptions.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  auto TempFileOrErr =
      createTempFile(Args, sys::path::filename(OutputFile), "spv");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back(File);

  if (Error Err = executeCommands(*LLVMToSPIRVPath, CmdArgs))
    return std::move(Err);
  return *TempFileOrErr;
}

/// Run sycl-post-link tool for SYCL offloading.
static Expected<std::vector<module_split::SplitModule>>
runSYCLPostLinkTool(StringRef LinkedFile, const ArgList &Args) {
  Expected<std::string> SYCLPostLinkPath =
      findProgram(Args, "sycl-post-link", {getMainExecutable("sycl-post-link")});
  if (!SYCLPostLinkPath)
    return SYCLPostLinkPath.takeError();

  auto TempFileOrErr =
      createTempFile(Args, sys::path::filename(OutputFile), "table");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*SYCLPostLinkPath);

  bool SpecConstsSupported =
      !Triple.isNVPTX() && !Triple.isAMDGCN() &&
      !Triple.isSPIRAOT() && !Triple.isNativeCPU();
  CmdArgs.push_back(SpecConstsSupported ? "-spec-const=native"
                                        : "-spec-const=emulation");

  CmdArgs.push_back("-properties");

  if (!Triple.isNVPTX() && !Triple.isAMDGPU())
    CmdArgs.push_back("-emit-only-kernels-as-entry-points");

  if (!Triple.isAMDGCN())
    CmdArgs.push_back("-emit-param-info");

  if (Triple.isNVPTX() || Triple.isAMDGCN() || Triple.isNativeCPU())
    CmdArgs.push_back("-emit-program-metadata");

  CmdArgs.push_back("-symbols");
  CmdArgs.push_back("-emit-exported-symbols");
  CmdArgs.push_back("-emit-imported-symbols");

  if (Triple.isSPIROrSPIRV())
    CmdArgs.push_back("-split-esimd");
  CmdArgs.push_back("-lower-esimd");

  StringRef UserOpts = Args.getLastArgValue(OPT_sycl_post_link_options_EQ);
  UserOpts.split(CmdArgs, " ", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back(LinkedFile);

  if (Error Err = executeCommands(*SYCLPostLinkPath, CmdArgs))
    return std::move(Err);

  return llvm::sycl_post_link::parseSplitModulesFromFile(*TempFileOrErr);
}

/// Performs the following steps:
/// 1. Link input device code (user code and SYCL device library code).
/// 2. Run sycl-post-link.
/// 3. Translate LLVM IR -> SPIR-V.
/// 4. AOT compile if needed.
/// 5. Pack into OffloadBinary and write to output.
Error runSYCLLink(ArrayRef<std::string> Files, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCL device linking");

  // Link all input bitcode files and SYCL device library files, if any.
  auto LinkedFile = linkDeviceCode(Files, Args);
  if (!LinkedFile)
    return LinkedFile.takeError();

  auto SplitModulesOrErr = runSYCLPostLinkTool(*LinkedFile, Args);
  if (!SplitModulesOrErr)
    return SplitModulesOrErr.takeError();
  auto &SplitModules = *SplitModulesOrErr;

  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  llvm::Triple::SubArchType SubArch = Triple.getSubArch();
  bool NeedAOTCompile =
      (SubArch == llvm::Triple::SPIRSubArch_gen) ||
      (SubArch == llvm::Triple::SPIRSubArch_x86_64);

  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(OutputFile, FD))
    return errorCodeToError(EC);
  llvm::raw_fd_ostream FS(FD, /*shouldClose=*/true);

  // LLVMContext needed only for symbol table extraction.
  LLVMContext C;

  for (size_t I = 0, E = SplitModules.size(); I != E; ++I) {
    auto SPVFileOrErr =
        runLLVMToSPIRVTranslation(SplitModules[I].ModuleFilePath, Args);
    if (!SPVFileOrErr)
      return SPVFileOrErr.takeError();
    StringRef ImageFile = *SPVFileOrErr;

    std::string AOTFile;
    if (NeedAOTCompile) {
      StringRef Stem = OutputFile.rsplit('.').first;
      AOTFile = (Stem + "_" + Twine(I) + ".out").str();
      if (Error Err = runAOTCompile(ImageFile, AOTFile, Args))
        return Err;
      ImageFile = AOTFile;
    }

    // Build symbol table from the bitcode module.
    auto ModOrErr = getBitcodeModule(SplitModules[I].ModuleFilePath, C);
    if (!ModOrErr)
      return ModOrErr.takeError();
    SmallString<0> SymbolData;
    for (Function &F : **ModOrErr) {
      if (isKernel(F)) {
        SymbolData.append(F.getName());
        SymbolData.push_back('\0');
      }
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(ImageFile);
    if (std::error_code EC = FileOrErr.getError())
      return createFileError(ImageFile, EC);

    OffloadingImage TheImage{};
    TheImage.TheImageKind = IMG_None;
    TheImage.TheOffloadKind = OFK_SYCL;
    TheImage.StringData["triple"] =
        Args.MakeArgString(Args.getLastArgValue(OPT_triple_EQ));
    TheImage.StringData["arch"] =
        Args.MakeArgString(Args.getLastArgValue(OPT_arch_EQ));
    TheImage.StringData["symbols"] = SymbolData;

    if (!NeedAOTCompile) {
      TheImage.StringData["compile-opts"] =
          Args.MakeArgString(SplitModules[I].CompileOptions);
      TheImage.StringData["link-opts"] =
          Args.MakeArgString(SplitModules[I].LinkOptions);
    }
    TheImage.Image = std::move(*FileOrErr);

    llvm::SmallString<0> Buffer = OffloadBinary::write(TheImage);
    if (Buffer.size() % OffloadBinary::getAlignment() != 0)
      return createStringError("Offload binary has invalid size alignment");
    FS << Buffer;
  }
  return Error::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::errs() << "[DEBUG] clang-sycl-linker called with args: "
               << llvm::join(argv, argv + argc, " ") << "\n";
  InitLLVM X(argc, argv);
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  Executable = argv[0];
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  const OptTable &Tbl = getOptTable();
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  auto Args = Tbl.parseArgs(argc, argv, OPT_INVALID, Saver, [&](StringRef Err) {
    reportError(createStringError(inconvertibleErrorCode(), Err));
  });

  if (Args.hasArg(OPT_help) || Args.hasArg(OPT_help_hidden)) {
    Tbl.printHelp(
        outs(), "clang-sycl-linker [options] <options to sycl link steps>",
        "A utility that wraps around several steps required to link SYCL "
        "device files.\n"
        "This enables LLVM IR linking, post-linking and code generation for "
        "SYCL targets.",
        Args.hasArg(OPT_help_hidden), Args.hasArg(OPT_help_hidden));
    return EXIT_SUCCESS;
  }

  if (Args.hasArg(OPT_version))
    printVersion(outs());

  Verbose = Args.hasArg(OPT_verbose);
  DryRun = Args.hasArg(OPT_dry_run);
  SaveTemps = Args.hasArg(OPT_save_temps);

  if (!Args.hasArg(OPT_o))
    reportError(createStringError("Output file must be specified"));
  OutputFile = Args.getLastArgValue(OPT_o);

  if (!Args.hasArg(OPT_triple_EQ))
    reportError(createStringError("Target triple must be specified"));

  if (Args.hasArg(OPT_spirv_dump_device_code_EQ)) {
    Arg *A = Args.getLastArg(OPT_spirv_dump_device_code_EQ);
    SmallString<128> Dir(A->getValue());
    if (Dir.empty())
      llvm::sys::path::native(Dir = "./");
    else
      Dir.append(llvm::sys::path::get_separator());
    SPIRVDumpDir = Dir;
  }

  // Get the input files to pass to the linking stage.
  auto FilesOrErr = getInput(Args);
  if (!FilesOrErr)
    reportError(FilesOrErr.takeError());

  // Run SYCL linking process on the generated inputs.
  if (Error Err = runSYCLLink(*FilesOrErr, Args))
    reportError(std::move(Err));

  // Remove the temporary files created.
  if (!Args.hasArg(OPT_save_temps))
    for (const auto &TempFile : TempFiles)
      if (std::error_code EC = sys::fs::remove(TempFile))
        reportError(createFileError(TempFile, EC));

  return EXIT_SUCCESS;
}