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

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
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
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::object;
using namespace clang;

/// Save intermediary results.
static bool SaveTemps = false;

/// Print commands/steps with arguments without executing.
static bool DryRun = false;

/// Print verbose output.
static bool Verbose = true;

/// Filename of the output being created.
static StringRef OutputFile;

/// Directory to dump SPIR-V IR if requested by user.
static SmallString<128> SPIRVDumpDir;

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
  SmallString<128> OutputFile;
  if (Args.hasArg(OPT_save_temps)) {
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
  for (const opt::Arg *Arg : Args.filtered(OPT_INPUT)) {
    std::optional<std::string> Filename = std::string(Arg->getValue());
    if (!Filename || !sys::fs::exists(*Filename) ||
        sys::fs::is_directory(*Filename))
      continue;
    file_magic Magic;
    if (auto EC = identify_magic(*Filename, Magic))
      return createStringError("Failed to open file " + *Filename);
    // TODO: Current use case involves LLVM IR bitcode files as input.
    // This will be extended to support SPIR-V IR files.
    switch (Magic) {
    case file_magic::unknown: 
      llvm::errs() << "[DEBUG] unknown file: " << *Filename << "\n"; break;
    case file_magic::bitcode: 
      llvm::errs() << "[DEBUG] bitcode file: " << *Filename << "\n"; break;
    case file_magic::clang_ast: 
      llvm::errs() << "[DEBUG] clang AST (PCH/PCM) file: " << *Filename << "\n"; break;
    case file_magic::archive: 
      llvm::errs() << "[DEBUG] archive file: " << *Filename << "\n"; break;
    case file_magic::elf: 
      llvm::errs() << "[DEBUG] ELF unknown type: " << *Filename << "\n"; break;
    case file_magic::elf_relocatable: 
      llvm::errs() << "[DEBUG] ELF relocatable object: " << *Filename << "\n"; break;
    case file_magic::elf_executable: 
      llvm::errs() << "[DEBUG] ELF executable: " << *Filename << "\n"; break;
    case file_magic::elf_shared_object: 
      llvm::errs() << "[DEBUG] ELF shared object: " << *Filename << "\n"; break;
    case file_magic::elf_core: 
      llvm::errs() << "[DEBUG] ELF core file: " << *Filename << "\n"; break;
    case file_magic::goff_object: 
      llvm::errs() << "[DEBUG] GOFF object file: " << *Filename << "\n"; break;
    case file_magic::macho_object: 
      llvm::errs() << "[DEBUG] Mach-O object: " << *Filename << "\n"; break;
    case file_magic::macho_executable: 
      llvm::errs() << "[DEBUG] Mach-O executable: " << *Filename << "\n"; break;
    case file_magic::macho_fixed_virtual_memory_shared_lib: 
      llvm::errs() << "[DEBUG] Mach-O shared lib (FVM): " << *Filename << "\n"; break;
    case file_magic::macho_core: 
      llvm::errs() << "[DEBUG] Mach-O core: " << *Filename << "\n"; break;
    case file_magic::macho_preload_executable: 
      llvm::errs() << "[DEBUG] Mach-O preload executable: " << *Filename << "\n"; break;
    case file_magic::macho_dynamically_linked_shared_lib: 
      llvm::errs() << "[DEBUG] Mach-O dynamically linked shared lib: " << *Filename << "\n"; break;
    case file_magic::macho_dynamic_linker: 
      llvm::errs() << "[DEBUG] Mach-O dynamic linker: " << *Filename << "\n"; break;
    case file_magic::macho_bundle: 
      llvm::errs() << "[DEBUG] Mach-O bundle: " << *Filename << "\n"; break;
    case file_magic::macho_dynamically_linked_shared_lib_stub: 
      llvm::errs() << "[DEBUG] Mach-O shared lib stub: " << *Filename << "\n"; break;
    case file_magic::macho_dsym_companion: 
      llvm::errs() << "[DEBUG] Mach-O dSYM companion: " << *Filename << "\n"; break;
    case file_magic::macho_kext_bundle: 
      llvm::errs() << "[DEBUG] Mach-O kext bundle: " << *Filename << "\n"; break;
    case file_magic::macho_universal_binary: 
      llvm::errs() << "[DEBUG] Mach-O universal binary: " << *Filename << "\n"; break;
    case file_magic::macho_file_set: 
      llvm::errs() << "[DEBUG] Mach-O file set: " << *Filename << "\n"; break;
    case file_magic::minidump: 
      llvm::errs() << "[DEBUG] Windows minidump: " << *Filename << "\n"; break;
    case file_magic::coff_cl_gl_object: 
      llvm::errs() << "[DEBUG] COFF cl.exe intermediate: " << *Filename << "\n"; break;
    case file_magic::coff_object: 
      llvm::errs() << "[DEBUG] COFF object: " << *Filename << "\n"; break;
    case file_magic::coff_import_library: 
      llvm::errs() << "[DEBUG] COFF import library: " << *Filename << "\n"; break;
    case file_magic::pecoff_executable: 
      llvm::errs() << "[DEBUG] PECOFF executable: " << *Filename << "\n"; break;
    case file_magic::windows_resource: 
      llvm::errs() << "[DEBUG] Windows resource file (.res): " << *Filename << "\n"; break;
    case file_magic::xcoff_object_32: 
      llvm::errs() << "[DEBUG] XCOFF 32-bit object: " << *Filename << "\n"; break;
    case file_magic::xcoff_object_64: 
      llvm::errs() << "[DEBUG] XCOFF 64-bit object: " << *Filename << "\n"; break;
    case file_magic::wasm_object: 
      llvm::errs() << "[DEBUG] WebAssembly object: " << *Filename << "\n"; break;
    case file_magic::pdb: 
      llvm::errs() << "[DEBUG] Windows PDB debug info: " << *Filename << "\n"; break;
    case file_magic::tapi_file: 
      llvm::errs() << "[DEBUG] TAPI stub file: " << *Filename << "\n"; break;
    case file_magic::cuda_fatbinary: 
      llvm::errs() << "[DEBUG] CUDA fatbinary: " << *Filename << "\n"; break;
    case file_magic::offload_binary: 
      llvm::errs() << "[DEBUG] LLVM offload object: " << *Filename << "\n"; break;
    case file_magic::dxcontainer_object: 
      llvm::errs() << "[DEBUG] DirectX container object: " << *Filename << "\n"; break;
    case file_magic::offload_bundle: 
      llvm::errs() << "[DEBUG] Clang offload bundle: " << *Filename << "\n"; break;
    case file_magic::offload_bundle_compressed: 
      llvm::errs() << "[DEBUG] Compressed Clang offload bundle: " << *Filename << "\n"; break;
    case file_magic::spirv_object: 
      llvm::errs() << "[DEBUG] SPIR-V object: " << *Filename << "\n"; break;
    default: 
      llvm::errs() << "[DEBUG] unhandled file type: " << *Filename << "\n"; break;
    }
    if(Magic == file_magic::elf_shared_object) {
      continue;
    }
    if(Magic == file_magic::spirv_object) {
      continue;
    }
    if (Magic != file_magic::bitcode)
      return createStringError("Unsupported file type");
    BitcodeFiles.push_back(*Filename);
  }
  return BitcodeFiles;
}

/// Handle cases where input file is a LLVM IR bitcode file.
/// When clang-sycl-linker is called via clang-linker-wrapper tool, input files
/// are LLVM IR bitcode files.
// TODO: Support SPIR-V IR files.
Expected<std::unique_ptr<Module>> getBitcodeModule(StringRef File,
                                                   LLVMContext &C) {
  SMDiagnostic Err;

  auto M = getLazyIRFileModule(File, Err, C);
  if (M)
    return std::move(M);
  return createStringError(Err.getMessage());
}

/// Gather all SYCL device library files that will be linked with input device
/// files.
/// The list of files and its location are passed from driver.
Expected<SmallVector<std::string>> getSYCLDeviceLibs(const ArgList &Args) {
  SmallVector<std::string> DeviceLibFiles;
  StringRef LibraryPath;
  if (Arg *A = Args.getLastArg(OPT_library_path_EQ))
    LibraryPath = A->getValue();
  if (Arg *A = Args.getLastArg(OPT_device_libs_EQ)) {
    if (A->getValues().size() == 0)
      return createStringError(
          inconvertibleErrorCode(),
          "Number of device library files cannot be zero.");
    for (StringRef Val : A->getValues()) {
      SmallString<128> LibName(LibraryPath);
      llvm::sys::path::append(LibName, Val);
      if (llvm::sys::fs::exists(LibName))
        DeviceLibFiles.push_back(std::string(LibName));
      else
        return createStringError(inconvertibleErrorCode(),
                                 "\'" + std::string(LibName) + "\'" +
                                     " SYCL device library file is not found.");
    }
  }
  return DeviceLibFiles;
}

/// Following tasks are performed:
/// 1. Link all SYCL device bitcode images into one image. Device linking is
/// performed using the linkInModule API.
/// 2. Gather all SYCL device library bitcode images.
/// 3. Link all the images gathered in Step 2 with the output of Step 1 using
/// linkInModule API. LinkOnlyNeeded flag is used.
Expected<StringRef> linkDeviceCode(ArrayRef<std::string> InputFiles,
                                   const ArgList &Args, LLVMContext &C) {
  llvm::TimeTraceScope TimeScope("SYCL link device code");

  assert(InputFiles.size() && "No inputs to link");

  auto LinkerOutput = std::make_unique<Module>("sycl-device-link", C);
  Linker L(*LinkerOutput);
  // Link SYCL device input files.
  for (auto &File : InputFiles) {
    auto ModOrErr = getBitcodeModule(File, C);
    if (!ModOrErr)
      return ModOrErr.takeError();
    if (L.linkInModule(std::move(*ModOrErr)))
      return createStringError("Could not link IR");
  }

  // Get all SYCL device library files, if any.
  auto SYCLDeviceLibFiles = getSYCLDeviceLibs(Args);
  if (!SYCLDeviceLibFiles)
    return SYCLDeviceLibFiles.takeError();

  // Link in SYCL device library files.
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  for (auto &File : *SYCLDeviceLibFiles) {
    auto LibMod = getBitcodeModule(File, C);
    if (!LibMod)
      return LibMod.takeError();
    if ((*LibMod)->getTargetTriple() == Triple) {
      unsigned Flags = Linker::Flags::LinkOnlyNeeded;
      if (L.linkInModule(std::move(*LibMod), Flags))
        return createStringError("Could not link IR");
    }
  }

  // Dump linked output for testing.
  if (Args.hasArg(OPT_print_linked_module))
    outs() << *LinkerOutput;

  // Create a new file to write the linked device file to.
  auto BitcodeOutput =
      createTempFile(Args, sys::path::filename(OutputFile), "bc");
  if (!BitcodeOutput)
    return BitcodeOutput.takeError();

  // Write the final output into 'BitcodeOutput' file.
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(*BitcodeOutput, FD))
    return errorCodeToError(EC);
  llvm::raw_fd_ostream OS(FD, true);
  WriteBitcodeToFile(*LinkerOutput, OS);

  if (Verbose) {
    std::string Inputs = llvm::join(InputFiles.begin(), InputFiles.end(), ", ");
    std::string LibInputs = llvm::join((*SYCLDeviceLibFiles).begin(),
                                       (*SYCLDeviceLibFiles).end(), ", ");
    errs() << formatv(
        "sycl-device-link: inputs: {0} libfiles: {1} output: {2}\n", Inputs,
        LibInputs, *BitcodeOutput);
  }

  return *BitcodeOutput;
}

/// Run Code Generation using LLVM backend.
/// \param 'File' The input LLVM IR bitcode file.
/// \param 'Args' encompasses all arguments required for linking device code and
/// will be parsed to generate options required to be passed into the backend.
/// \param 'OutputFile' The output file name.
/// \param 'C' The LLVM context.
static Error runCodeGen(StringRef File, const ArgList &Args,
                        StringRef OutputFile, LLVMContext &C) {
  llvm::TimeTraceScope TimeScope("Code generation");

  // Parse input module.
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(File, Err, C);
  if (!M)
    return createStringError(Err.getMessage());

  if (Error Err = M->materializeAll())
    return Err;

  Triple TargetTriple(Args.getLastArgValue(OPT_triple_EQ));

  M->setTargetTriple(TargetTriple);

  // Get a handle to a target backend.
  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M->getTargetTriple(), Msg);
  if (!T)
    return createStringError(Msg + ": " + M->getTargetTriple().str());

  // Allocate target machine.
  TargetOptions Options;
  std::optional<Reloc::Model> RM;
  std::optional<CodeModel::Model> CM;
  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(M->getTargetTriple(), /* CPU */ "",
                             /* Features */ "", Options, RM, CM));
  if (!TM)
    return createStringError("Could not allocate target machine!");

  // Set data layout if needed.
  if (M->getDataLayout().isDefault())
    M->setDataLayout(TM->createDataLayout());

  // Open output file for writing.
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(OutputFile, FD))
    return errorCodeToError(EC);
  auto OS = std::make_unique<llvm::raw_fd_ostream>(FD, true);

  legacy::PassManager CodeGenPasses;
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));
  if (TM->addPassesToEmitFile(CodeGenPasses, *OS, nullptr,
                              CodeGenFileType::ObjectFile))
    return createStringError("Failed to execute LLVM backend");
  CodeGenPasses.run(*M);

  if (Verbose)
    errs() << formatv("LLVM backend: input: {0}, output: {1}\n", File,
                      OutputFile);

  return Error::success();
}

/// Run AOT compilation for Intel CPU.
/// Calls opencl-aot tool to generate device code for the Intel OpenCL CPU
/// Runtime.
/// \param InputFile The input SPIR-V file.
/// \param OutputFile The output file name.
/// \param Args Encompasses all arguments required for linking and wrapping
/// device code and will be parsed to generate options required to be passed
/// into the SYCL AOT compilation step.
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
/// Calls ocloc tool to generate device code for the Intel Graphics Compute
/// Runtime.
/// \param InputFile The input SPIR-V file.
/// \param OutputFile The output file name.
/// \param Args Encompasses all arguments required for linking and wrapping
/// device code and will be parsed to generate options required to be passed
/// into the SYCL AOT compilation step.
static Error runAOTCompileIntelGPU(StringRef InputFile, StringRef OutputFile,
                                   const ArgList &Args) {
  SmallVector<StringRef, 8> CmdArgs;
  Expected<std::string> OclocPath =
      findProgram(Args, "ocloc", {getMainExecutable("ocloc")});
  if (!OclocPath)
    return OclocPath.takeError();

  CmdArgs.push_back(*OclocPath);
  // The next line prevents ocloc from modifying the image name
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
/// \param InputFile The input SPIR-V file.
/// \param OutputFile The output file name.
/// \param Args Encompasses all arguments required for linking and wrapping
/// device code and will be parsed to generate options required to be passed
/// into the SYCL AOT compilation step.
static Error runAOTCompile(StringRef InputFile, StringRef OutputFile,
                           const ArgList &Args) {
  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);
  OffloadArch OffloadArch = StringToOffloadArch(Arch);
  if (IsIntelGPUOffloadArch(OffloadArch))
    return runAOTCompileIntelGPU(InputFile, OutputFile, Args);
  if (IsIntelCPUOffloadArch(OffloadArch))
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
/// NOTE: Any changes made here should be reflected in the similarly named
/// function in clang/lib/Driver/ToolChains/Clang.cpp.
static void
getTripleBasedSPIRVTransOpts(const ArgList &Args,
                             SmallVector<StringRef, 8> &TranslatorArgs,
                             const llvm::Triple Triple) {
  bool IsCPU = Triple.isSPIR() &&
               Triple.getSubArch() == llvm::Triple::SPIRSubArch_x86_64;
  TranslatorArgs.push_back(
      "-spirv-debug-info-version=nonsemantic-shader-200");
  std::string UnknownIntrinsics(
      "-spirv-allow-unknown-intrinsics=llvm.genx.");
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
/// Converts 'File' from LLVM bitcode to SPIR-V format using llvm-spirv tool.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// llvm-spirv tool.
static Expected<StringRef> runLLVMToSPIRVTranslation(StringRef File,
                                                      const ArgList &Args) {
  Expected<std::string> LLVMToSPIRVPath =
      findProgram(Args, "llvm-spirv",
                  {getMainExecutable("llvm-spirv")});
  if (!LLVMToSPIRVPath)
    return LLVMToSPIRVPath.takeError();

  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*LLVMToSPIRVPath);

  // Add triple-based translation options.
  getTripleBasedSPIRVTransOpts(Args, CmdArgs, Triple);

  // Forward any user-supplied llvm-spirv options.
  StringRef LLVMToSPIRVOptions =
      Args.getLastArgValue(OPT_llvm_spirv_options_EQ);
  LLVMToSPIRVOptions.split(CmdArgs, " ", /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);

  // Create output file.
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
/// 'InputFiles' is the list of input LLVM IR files.
/// 'Args' encompasses all arguments required for linking and wrapping device
/// code and will be parsed to generate options required to be passed into the
/// sycl-post-link tool.
/// 'IsDevicePassedWithSyclTargetBackend' indicates whether the device
/// architecture is already specified through -Xsycl-target-backend=spir64_gen
/// "-device <arch>" format.
static Expected<std::vector<module_split::SplitModule>>
runSYCLPostLinkTool(StringRef LinkedFile, const ArgList &Args) {
  Expected<std::string> SYCLPostLinkPath =
      findProgram(Args, "sycl-post-link",
                  {getMainExecutable("sycl-post-link")});
  if (!SYCLPostLinkPath)
    return SYCLPostLinkPath.takeError();

  // Create output table file.
  auto TempFileOrErr =
      createTempFile(Args, sys::path::filename(OutputFile), "table");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  SmallVector<StringRef, 8> CmdArgs;
  CmdArgs.push_back(*SYCLPostLinkPath);

  // Add specialization constant mode based on triple.
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

  if (Triple.isSPIROrSPIRV()) {
    CmdArgs.push_back("-split-esimd");
  }
  CmdArgs.push_back("-lower-esimd");

  // Forward any user-supplied post-link options.
  StringRef UserOpts =
      Args.getLastArgValue(OPT_sycl_post_link_options_EQ);
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
/// 2. Run SPIR-V code generation.
Error runSYCLLink(ArrayRef<std::string> Files, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("SYCL device linking");

  LLVMContext C;

  // Link all input bitcode files and SYCL device library files, if any.
  auto LinkedFile = linkDeviceCode(Files, Args, C);
  if (!LinkedFile)
    return LinkedFile.takeError();

  // TODO: SYCL post link functionality involves device code splitting and will
  // result in multiple bitcode codes.
  // The following lines are placeholders to represent multiple files and will
  // be refactored once SYCL post link support is available.
  auto SplitModulesOrErr = runSYCLPostLinkTool(*LinkedFile, Args);
  if (!SplitModulesOrErr)
    return SplitModulesOrErr.takeError();
  auto &SplitModules = *SplitModulesOrErr;

  // Determine if AOT compilation is needed.
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  llvm::Triple::SubArchType SubArch = Triple.getSubArch();
  bool NeedAOTCompile =
      (SubArch == llvm::Triple::SPIRSubArch_gen) ||
      (SubArch == llvm::Triple::SPIRSubArch_x86_64);

  // Open the final output file.
  int FD = -1;
  if (std::error_code EC =
          sys::fs::openFileForWrite(OutputFile, FD))
    return errorCodeToError(EC);
  llvm::raw_fd_ostream FS(FD, /*shouldClose=*/true);

  for (size_t I = 0, E = SplitModules.size(); I != E; ++I) {
    // Step 4: Translate LLVM IR -> SPIR-V.
    auto SPVFileOrErr =
        runLLVMToSPIRVTranslation(SplitModules[I].ModuleFilePath, Args);
    if (!SPVFileOrErr)
      return SPVFileOrErr.takeError();
    StringRef ImageFile = *SPVFileOrErr;

    // Step 5: AOT compile if needed (Intel GPU or CPU).
    std::string AOTFile;
    if (NeedAOTCompile) {
      StringRef Stem = OutputFile.rsplit('.').first;
      AOTFile = (Stem + "_" + Twine(I) + ".out").str();
      if (Error Err = runAOTCompile(ImageFile, AOTFile, Args))
        return Err;
      ImageFile = AOTFile;
    }

    // Build symbol table from the bitcode module.
    auto ModOrErr =
        getBitcodeModule(SplitModules[I].ModuleFilePath, C);
    if (!ModOrErr)
      return ModOrErr.takeError();
    SmallString<0> SymbolData;
    for (Function &F : **ModOrErr) {
      if (isKernel(F)) {
        SymbolData.append(F.getName());
        SymbolData.push_back('\0');
      }
    }

    // Read the final image (SPIR-V or AOT binary).
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(ImageFile);
    if (std::error_code EC = FileOrErr.getError())
      return createFileError(ImageFile, EC);

    // Pack into OffloadBinary.
    OffloadingImage TheImage{};
    TheImage.TheImageKind = IMG_None;
    TheImage.TheOffloadKind = OFK_SYCL;
    TheImage.StringData["triple"] =
        Args.MakeArgString(Args.getLastArgValue(OPT_triple_EQ));
    TheImage.StringData["arch"] =
        Args.MakeArgString(Args.getLastArgValue(OPT_arch_EQ));
    TheImage.StringData["symbols"] = SymbolData;

    // For JIT path, carry compile/link options.
    if (!NeedAOTCompile) {
      TheImage.StringData["compile-opts"] =
          Args.MakeArgString(SplitModules[I].CompileOptions);
      TheImage.StringData["link-opts"] =
          Args.MakeArgString(SplitModules[I].LinkOptions);
    }
    TheImage.Image = std::move(*FileOrErr);

    llvm::SmallString<0> Buffer = OffloadBinary::write(TheImage);
    if (Buffer.size() % OffloadBinary::getAlignment() != 0)
      return createStringError(
          "Offload binary has invalid size alignment");
    FS << Buffer;
  }
  return Error::success();
}

} // namespace

int main(int argc, char **argv) {
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
