//===--- SYCL.cpp - SYCL Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SYCL.h"
#include "CommonArgs.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <sstream>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

SYCLInstallationDetector::SYCLInstallationDetector(const Driver &D)
    : D(D), InstallationCandidates() {
  InstallationCandidates.emplace_back(D.Dir + "/..");
}

void SYCLInstallationDetector::getSYCLDeviceLibPath(
    llvm::SmallVector<llvm::SmallString<128>, 4> &DeviceLibPaths) const {
  for (const auto &IC : InstallationCandidates) {
    llvm::SmallString<128> InstallLibPath(IC.str());
    InstallLibPath.append("/lib");
    DeviceLibPaths.emplace_back(InstallLibPath);
  }

  DeviceLibPaths.emplace_back(D.SysRoot + "/lib");
}

void SYCLInstallationDetector::print(llvm::raw_ostream &OS) const {
  if (!InstallationCandidates.size())
    return;
  OS << "SYCL Installation Candidates: \n";
  for (const auto &IC : InstallationCandidates) {
    OS << IC << "\n";
  }
}

static void addFPGATimingDiagnostic(std::unique_ptr<Command> &Cmd,
                                    Compilation &C) {
  const char *Msg = C.getArgs().MakeArgString(
      "The FPGA image generated during this compile contains timing violations "
      "and may produce functional errors if used. Refer to the Intel oneAPI "
      "DPC++ FPGA Optimization Guide section on Timing Failures for more "
      "information.");
  Cmd->addDiagForErrorCode(/*ErrorCode*/ 42, Msg);
  Cmd->addExitForErrorCode(/*ErrorCode*/ 42, false);
}

void SYCL::constructLLVMForeachCommand(Compilation &C, const JobAction &JA,
                                       std::unique_ptr<Command> InputCommand,
                                       const InputInfoList &InputFiles,
                                       const InputInfo &Output, const Tool *T,
                                       StringRef Increment, StringRef Ext,
                                       StringRef ParallelJobs) {
  // Construct llvm-foreach command.
  // The llvm-foreach command looks like this:
  // llvm-foreach --in-file-list=a.list --in-replace='{}' -- echo '{}'
  ArgStringList ForeachArgs;
  std::string OutputFileName(T->getToolChain().getInputFilename(Output));
  ForeachArgs.push_back(C.getArgs().MakeArgString("--out-ext=" + Ext));
  for (auto &I : InputFiles) {
    std::string Filename(T->getToolChain().getInputFilename(I));
    ForeachArgs.push_back(
        C.getArgs().MakeArgString("--in-file-list=" + Filename));
    ForeachArgs.push_back(
        C.getArgs().MakeArgString("--in-replace=" + Filename));
  }

  ForeachArgs.push_back(
      C.getArgs().MakeArgString("--out-file-list=" + OutputFileName));
  ForeachArgs.push_back(
      C.getArgs().MakeArgString("--out-replace=" + OutputFileName));
  if (!Increment.empty())
    ForeachArgs.push_back(
        C.getArgs().MakeArgString("--out-increment=" + Increment));
  if (!ParallelJobs.empty())
    ForeachArgs.push_back(C.getArgs().MakeArgString("--jobs=" + ParallelJobs));

  if (C.getDriver().isSaveTempsEnabled()) {
    SmallString<128> OutputDirName;
    if (C.getDriver().isSaveTempsObj()) {
      OutputDirName =
          T->getToolChain().GetFilePath(OutputFileName.c_str()).c_str();
      llvm::sys::path::remove_filename(OutputDirName);
    }
    // Use the current dir if the `GetFilePath` returned en empty string, which
    // is the case when the `OutputFileName` does not contain any directory
    // information, or if in CWD mode. This is necessary for `llvm-foreach`, as
    // it would disregard the parameter without it. Otherwise append separator.
    if (OutputDirName.empty())
      llvm::sys::path::native(OutputDirName = "./");
    else
      OutputDirName.append(llvm::sys::path::get_separator());
    ForeachArgs.push_back(
        C.getArgs().MakeArgString("--out-dir=" + OutputDirName));
  }

  // If fsycl-dump-device-code is passed, put the PTX files
  // into the path provided in fsycl-dump-device-code.
  if (T->getToolChain().getTriple().isNVPTX() &&
      C.getDriver().isDumpDeviceCodeEnabled() && Ext.equals("s")) {
    SmallString<128> OutputDir;

    Arg *DumpDeviceCodeArg =
        C.getArgs().getLastArg(options::OPT_fsycl_dump_device_code_EQ);

    OutputDir = (DumpDeviceCodeArg ? DumpDeviceCodeArg->getValue() : "");

    // If the output directory path is empty, put the PTX files in the
    // current directory.
    if (OutputDir.empty())
      llvm::sys::path::native(OutputDir = "./");
    else
      OutputDir.append(llvm::sys::path::get_separator());
    ForeachArgs.push_back(C.getArgs().MakeArgString("--out-dir=" + OutputDir));
  }

  ForeachArgs.push_back(C.getArgs().MakeArgString("--"));
  ForeachArgs.push_back(
      C.getArgs().MakeArgString(InputCommand->getExecutable()));

  for (auto &Arg : InputCommand->getArguments())
    ForeachArgs.push_back(Arg);

  SmallString<128> ForeachPath(C.getDriver().Dir);
  llvm::sys::path::append(ForeachPath, "llvm-foreach");
  const char *Foreach = C.getArgs().MakeArgString(ForeachPath);

  auto Cmd = std::make_unique<Command>(JA, *T, ResponseFileSupport::None(),
                                       Foreach, ForeachArgs, std::nullopt);
  // FIXME: Add the FPGA specific timing diagnostic to the foreach call.
  // The foreach call obscures the return codes from the tool it is calling
  // to the compiler itself.
  addFPGATimingDiagnostic(Cmd, C);
  C.addCommand(std::move(Cmd));
}

bool SYCL::shouldDoPerObjectFileLinking(const Compilation &C) {
  return !C.getArgs().hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                              /*default=*/true);
}

// Return whether to use native bfloat16 library.
static bool selectBfloatLibs(const llvm::Triple &Triple, const Compilation &C,
                             bool &UseNative) {
  const llvm::opt::ArgList &Args = C.getArgs();
  bool NeedLibs = false;

  // spir64 target is actually JIT compilation, so we defer selection of
  // bfloat16 libraries to runtime. For AOT we need libraries, but skip
  // for Nvidia.
  NeedLibs =
      Triple.getSubArch() != llvm::Triple::NoSubArch && !Triple.isNVPTX();
  UseNative = false;
  if (NeedLibs && Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen &&
      C.hasOffloadToolChain<Action::OFK_SYCL>()) {
    ArgStringList TargArgs;
    auto ToolChains = C.getOffloadToolChains<Action::OFK_SYCL>();
    // Match up the toolchain with the incoming Triple so we are grabbing the
    // expected arguments to scrutinize.
    for (auto TI = ToolChains.first, TE = ToolChains.second; TI != TE; ++TI) {
      llvm::Triple SYCLTriple = TI->second->getTriple();
      if (SYCLTriple == Triple) {
        const toolchains::SYCLToolChain *SYCLTC =
            static_cast<const toolchains::SYCLToolChain *>(TI->second);
        SYCLTC->TranslateBackendTargetArgs(Triple, Args, TargArgs);
        break;
      }
    }

    auto checkBF = [](StringRef Device) {
      return Device.starts_with("pvc") || Device.starts_with("ats");
    };

    std::string Params;
    for (const auto &Arg : TargArgs) {
      Params += " ";
      Params += Arg;
    }
    size_t DevicesPos = Params.find("-device ");
    UseNative = false;
    if (DevicesPos != std::string::npos) {
      UseNative = true;
      std::istringstream Devices(Params.substr(DevicesPos + 8));
      for (std::string S; std::getline(Devices, S, ',');)
        UseNative &= checkBF(S);
    }
  }
  return NeedLibs;
}

SmallVector<std::string, 8>
SYCL::getDeviceLibraries(const Compilation &C, const llvm::Triple &TargetTriple,
                         bool IsSpirvAOT) {
  SmallVector<std::string, 8> LibraryList;
  const llvm::opt::ArgList &Args = C.getArgs();

  struct DeviceLibOptInfo {
    StringRef DeviceLibName;
    StringRef DeviceLibOption;
  };

  bool NoDeviceLibs = false;
  // Currently, all SYCL device libraries will be linked by default. Linkage
  // of "internal" libraries cannot be affected via -fno-sycl-device-lib.
  llvm::StringMap<bool> DeviceLibLinkInfo = {
      {"libc", true},          {"libm-fp32", true},   {"libm-fp64", true},
      {"libimf-fp32", true},   {"libimf-fp64", true}, {"libimf-bf16", true},
      {"libm-bfloat16", true}, {"internal", true}};
  if (Arg *A = Args.getLastArg(options::OPT_fsycl_device_lib_EQ,
                               options::OPT_fno_sycl_device_lib_EQ)) {
    if (A->getValues().size() == 0)
      C.getDriver().Diag(diag::warn_drv_empty_joined_argument)
          << A->getAsString(Args);
    else {
      if (A->getOption().matches(options::OPT_fno_sycl_device_lib_EQ))
        NoDeviceLibs = true;

      for (StringRef Val : A->getValues()) {
        if (Val == "all") {
          for (const auto &K : DeviceLibLinkInfo.keys())
            DeviceLibLinkInfo[K] =
                true && (!NoDeviceLibs || K.equals("internal"));
          break;
        }
        auto LinkInfoIter = DeviceLibLinkInfo.find(Val);
        if (LinkInfoIter == DeviceLibLinkInfo.end() || Val.equals("internal")) {
          // TODO: Move the diagnostic to the SYCL section of
          // Driver::CreateOffloadingDeviceToolChains() to minimize code
          // duplication.
          C.getDriver().Diag(diag::err_drv_unsupported_option_argument)
              << A->getSpelling() << Val;
        }
        DeviceLibLinkInfo[Val] = true && !NoDeviceLibs;
      }
    }
  }
  StringRef LibSuffix =
      C.getDefaultToolChain().getTriple().isWindowsMSVCEnvironment() ? ".obj"
                                                                     : ".o";
  using SYCLDeviceLibsList = SmallVector<DeviceLibOptInfo, 5>;

  const SYCLDeviceLibsList SYCLDeviceWrapperLibs = {
      {"libsycl-crt", "libc"},
      {"libsycl-complex", "libm-fp32"},
      {"libsycl-complex-fp64", "libm-fp64"},
      {"libsycl-cmath", "libm-fp32"},
      {"libsycl-cmath-fp64", "libm-fp64"},
#if defined(_WIN32)
      {"libsycl-msvc-math", "libm-fp32"},
#endif
      {"libsycl-imf", "libimf-fp32"},
      {"libsycl-imf-fp64", "libimf-fp64"},
      {"libsycl-imf-bf16", "libimf-bf16"}};
  // For AOT compilation, we need to link sycl_device_fallback_libs as
  // default too.
  const SYCLDeviceLibsList SYCLDeviceFallbackLibs = {
      {"libsycl-fallback-cassert", "libc"},
      {"libsycl-fallback-cstring", "libc"},
      {"libsycl-fallback-complex", "libm-fp32"},
      {"libsycl-fallback-complex-fp64", "libm-fp64"},
      {"libsycl-fallback-cmath", "libm-fp32"},
      {"libsycl-fallback-cmath-fp64", "libm-fp64"},
      {"libsycl-fallback-imf", "libimf-fp32"},
      {"libsycl-fallback-imf-fp64", "libimf-fp64"},
      {"libsycl-fallback-imf-bf16", "libimf-bf16"}};
  const SYCLDeviceLibsList SYCLDeviceBfloat16FallbackLib = {
      {"libsycl-fallback-bfloat16", "libm-bfloat16"}};
  const SYCLDeviceLibsList SYCLDeviceBfloat16NativeLib = {
      {"libsycl-native-bfloat16", "libm-bfloat16"}};
  // ITT annotation libraries are linked in separately whenever the device
  // code instrumentation is enabled.
  const SYCLDeviceLibsList SYCLDeviceAnnotationLibs = {
      {"libsycl-itt-user-wrappers", "internal"},
      {"libsycl-itt-compiler-wrappers", "internal"},
      {"libsycl-itt-stubs", "internal"}};
#if !defined(_WIN32)
  const SYCLDeviceLibsList SYCLDeviceSanitizerLibs = {
      {"libsycl-sanitizer", "internal"}};
#endif

  auto addLibraries = [&](const SYCLDeviceLibsList &LibsList) {
    for (const DeviceLibOptInfo &Lib : LibsList) {
      if (!DeviceLibLinkInfo[Lib.DeviceLibOption])
        continue;
      SmallString<128> LibName(Lib.DeviceLibName);
      llvm::sys::path::replace_extension(LibName, LibSuffix);
      LibraryList.push_back(Args.MakeArgString(LibName));
    }
  };

  addLibraries(SYCLDeviceWrapperLibs);
  if (IsSpirvAOT || TargetTriple.isNVPTX())
    addLibraries(SYCLDeviceFallbackLibs);

  bool NativeBfloatLibs;
  bool NeedBfloatLibs = selectBfloatLibs(TargetTriple, C, NativeBfloatLibs);
  if (NeedBfloatLibs) {
    // Add native or fallback bfloat16 library.
    if (NativeBfloatLibs)
      addLibraries(SYCLDeviceBfloat16NativeLib);
    else
      addLibraries(SYCLDeviceBfloat16FallbackLib);
  }

  if (Args.hasFlag(options::OPT_fsycl_instrument_device_code,
                   options::OPT_fno_sycl_instrument_device_code, true))
    addLibraries(SYCLDeviceAnnotationLibs);

#if !defined(_WIN32)
  if (Arg *A = Args.getLastArg(options::OPT_fsanitize_EQ,
                               options::OPT_fno_sanitize_EQ)) {
    if (A->getOption().matches(options::OPT_fsanitize_EQ) &&
        A->getValues().size() == 1) {
      std::string SanitizeVal = A->getValue();
      if (SanitizeVal == "address")
        addLibraries(SYCLDeviceSanitizerLibs);
    }
  } else {
    // User can pass -fsanitize=address to device compiler via
    // -Xsycl-target-frontend, sanitize device library must be
    // linked with user's device image if so.
    bool IsDeviceAsanEnabled = false;
    auto SyclFEArg = Args.getAllArgValues(options::OPT_Xsycl_frontend);
    IsDeviceAsanEnabled = (std::count(SyclFEArg.begin(), SyclFEArg.end(),
                                      "-fsanitize=address") > 0);
    if (!IsDeviceAsanEnabled) {
      auto SyclFEArgEq = Args.getAllArgValues(options::OPT_Xsycl_frontend_EQ);
      IsDeviceAsanEnabled = (std::count(SyclFEArgEq.begin(), SyclFEArgEq.end(),
                                        "-fsanitize=address") > 0);
    }

    // User can also enable asan for SYCL device via -Xarch_device option.
    if (!IsDeviceAsanEnabled) {
      auto DeviceArchVals = Args.getAllArgValues(options::OPT_Xarch_device);
      for (auto DArchVal : DeviceArchVals) {
        if (DArchVal.find("-fsanitize=address") != std::string::npos) {
          IsDeviceAsanEnabled = true;
          break;
        }
      }
    }

    if (IsDeviceAsanEnabled)
      addLibraries(SYCLDeviceSanitizerLibs);
  }
#endif
  return LibraryList;
}

// The list should match pre-built SYCL device library files located in
// compiler package. Once we add or remove any SYCL device library files,
// the list should be updated accordingly.
static llvm::SmallVector<StringRef, 16> SYCLDeviceLibList {
  "bfloat16", "crt", "cmath", "cmath-fp64", "complex", "complex-fp64",
#if defined(_WIN32)
      "msvc-math",
#else
      "sanitizer",
#endif
      "imf", "imf-fp64", "itt-compiler-wrappers", "itt-stubs",
      "itt-user-wrappers", "fallback-cassert", "fallback-cstring",
      "fallback-cmath", "fallback-cmath-fp64", "fallback-complex",
      "fallback-complex-fp64", "fallback-imf", "fallback-imf-fp64",
      "fallback-imf-bf16", "fallback-bfloat16", "native-bfloat16"
};

const char *SYCL::Linker::constructLLVMLinkCommand(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const ArgList &Args, StringRef SubArchName, StringRef OutputFilePrefix,
    const InputInfoList &InputFiles) const {
  // Split inputs into libraries which have 'archive' type and other inputs
  // which can be either objects or list files. Object files are linked together
  // in a usual way, but the libraries/list files need to be linked differently.
  // We need to fetch only required symbols from the libraries. With the current
  // llvm-link command line interface that can be achieved with two step
  // linking: at the first step we will link objects into an intermediate
  // partially linked image which on the second step will be linked with the
  // libraries with --only-needed option.
  ArgStringList Opts;
  ArgStringList Objs;
  ArgStringList Libs;
  // Add the input bc's created by compile step.
  // When offloading, the input file(s) could be from unbundled partially
  // linked archives.  The unbundled information is a list of files and not
  // an actual object/archive.  Take that list and pass those to the linker
  // instead of the original object.
  if (JA.isDeviceOffloading(Action::OFK_SYCL)) {
    bool IsRDC = !shouldDoPerObjectFileLinking(C);
    const bool IsSYCLNativeCPU = isSYCLNativeCPU(this->getToolChain());
    auto isNoRDCDeviceCodeLink = [&](const InputInfo &II) {
      if (IsRDC)
        return false;
      if (II.getType() != clang::driver::types::TY_LLVM_BC)
        return false;
      if (InputFiles.size() != 2)
        return false;
      return &II == &InputFiles[1];
    };
    auto isSYCLDeviceLib = [&](const InputInfo &II) {
      const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
      StringRef LibPostfix = ".o";
      if (isNoRDCDeviceCodeLink(II))
        LibPostfix = ".bc";
      else if (HostTC->getTriple().isWindowsMSVCEnvironment() &&
               C.getDriver().IsCLMode())
        LibPostfix = ".obj";
      std::string FileName = this->getToolChain().getInputFilename(II);
      StringRef InputFilename = llvm::sys::path::filename(FileName);
      const bool IsNVPTX = this->getToolChain().getTriple().isNVPTX();
      if (IsNVPTX || IsSYCLNativeCPU) {
        // Linking SYCL Device libs requires libclc as well as libdevice
        if ((InputFilename.find("libspirv") != InputFilename.npos ||
             InputFilename.find("libdevice") != InputFilename.npos))
          return true;
        if (IsNVPTX)
          LibPostfix = ".cubin";
      }
      StringRef LibSyclPrefix("libsycl-");
      if (!InputFilename.starts_with(LibSyclPrefix) ||
          !InputFilename.ends_with(LibPostfix) ||
          (InputFilename.count('-') < 2))
        return false;
      // Skip the prefix "libsycl-"
      std::string PureLibName =
          InputFilename.substr(LibSyclPrefix.size()).str();
      if (isNoRDCDeviceCodeLink(II)) {
        // Skip the final - until the . because we linked all device libs into a
        // single BC in a previous action so we have a temp file name.
        auto FinalDashPos = PureLibName.find_last_of('-');
        auto DotPos = PureLibName.find_last_of('.');
        assert((FinalDashPos != std::string::npos &&
                DotPos != std::string::npos) &&
               "Unexpected filename");
        PureLibName =
            PureLibName.substr(0, FinalDashPos) + PureLibName.substr(DotPos);
      }
      for (const auto &L : SYCLDeviceLibList) {
        if (StringRef(PureLibName).starts_with(L))
          return true;
      }
      return false;
    };
    size_t InputFileNum = InputFiles.size();
    bool LinkSYCLDeviceLibs = (InputFileNum >= 2);
    LinkSYCLDeviceLibs = LinkSYCLDeviceLibs && !isSYCLDeviceLib(InputFiles[0]);
    for (size_t Idx = 1; Idx < InputFileNum; ++Idx)
      LinkSYCLDeviceLibs =
          LinkSYCLDeviceLibs && isSYCLDeviceLib(InputFiles[Idx]);
    // Go through the Inputs to the link.  When a listfile is encountered, we
    // know it is an unbundled generated list.
    if (LinkSYCLDeviceLibs) {
      Opts.push_back("-only-needed");
    }
    for (const auto &II : InputFiles) {
      std::string FileName = getToolChain().getInputFilename(II);
      if (II.getType() == types::TY_Tempfilelist) {
        if (IsRDC) {
          // Pass the unbundled list with '@' to be processed.
          Libs.push_back(C.getArgs().MakeArgString("@" + FileName));
        } else {
          assert(InputFiles.size() == 2 &&
                 "Unexpected inputs for no-RDC with temp file list");
          // If we're in no-RDC mode and the input is a temp file list,
          // we want to link multiple object files each against device libs,
          // so we should consider this input as an object and not pass '@'.
          Objs.push_back(C.getArgs().MakeArgString(FileName));
        }
      } else if (II.getType() == types::TY_Archive && !LinkSYCLDeviceLibs) {
        Libs.push_back(C.getArgs().MakeArgString(FileName));
      } else
        Objs.push_back(C.getArgs().MakeArgString(FileName));
    }
  } else
    for (const auto &II : InputFiles)
      Objs.push_back(
          C.getArgs().MakeArgString(getToolChain().getInputFilename(II)));

  // Get llvm-link path.
  SmallString<128> ExecPath(C.getDriver().Dir);
  llvm::sys::path::append(ExecPath, "llvm-link");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);

  auto AddLinkCommand = [this, &C, &JA, Exec](const char *Output,
                                              const ArgStringList &Inputs,
                                              const ArgStringList &Options) {
    ArgStringList CmdArgs;
    llvm::copy(Options, std::back_inserter(CmdArgs));
    llvm::copy(Inputs, std::back_inserter(CmdArgs));
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output);
    // TODO: temporary workaround for a problem with warnings reported by
    // llvm-link when driver links LLVM modules with empty modules
    CmdArgs.push_back("--suppress-warnings");
    C.addCommand(std::make_unique<Command>(JA, *this,
                                           ResponseFileSupport::AtFileUTF8(),
                                           Exec, CmdArgs, std::nullopt));
  };

  // Add an intermediate output file.
  const char *OutputFileName =
      C.getArgs().MakeArgString(getToolChain().getInputFilename(Output));

  if (Libs.empty())
    AddLinkCommand(OutputFileName, Objs, Opts);
  else {
    assert(Opts.empty() && "unexpected options");

    // Linker will be invoked twice if inputs contain libraries. First time we
    // will link input objects into an intermediate temporary file, and on the
    // second invocation intermediate temporary object will be linked with the
    // libraries, but now only required symbols will be added to the final
    // output.
    std::string TempFile =
        C.getDriver().GetTemporaryPath(OutputFilePrefix.str() + "-link", "bc");
    const char *LinkOutput = C.addTempFile(C.getArgs().MakeArgString(TempFile));
    AddLinkCommand(LinkOutput, Objs, {});

    // Now invoke linker for the second time to link required symbols from the
    // input libraries.
    ArgStringList LinkInputs{LinkOutput};
    llvm::copy(Libs, std::back_inserter(LinkInputs));
    AddLinkCommand(OutputFileName, LinkInputs, {"--only-needed"});
  }
  return OutputFileName;
}

void SYCL::Linker::constructLlcCommand(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const char *InputFileName) const {
  // Construct llc command.
  // The output is an object file.
  ArgStringList LlcArgs{"-filetype=obj", "-o", Output.getFilename(),
                        InputFileName};
  SmallString<128> LlcPath(C.getDriver().Dir);
  llvm::sys::path::append(LlcPath, "llc");
  const char *Llc = C.getArgs().MakeArgString(LlcPath);
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileUTF8(), Llc,
                                         LlcArgs, std::nullopt));
}

// For SYCL the inputs of the linker job are SPIR-V binaries and output is
// a single SPIR-V binary.  Input can also be bitcode when specified by
// the user.
void SYCL::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {

  assert((getToolChain().getTriple().isSPIR() ||
          getToolChain().getTriple().isNVPTX() ||
          getToolChain().getTriple().isAMDGCN() || isSYCLNativeCPU(Args)) &&
         "Unsupported target");

  std::string SubArchName =
      std::string(getToolChain().getTriple().getArchName());

  // Prefix for temporary file name.
  std::string Prefix = std::string(llvm::sys::path::stem(SubArchName));

  // For CUDA, we want to link all BC files before resuming the normal
  // compilation path
  if (getToolChain().getTriple().isNVPTX() ||
      getToolChain().getTriple().isAMDGCN()) {
    InputInfoList NvptxInputs;
    for (const auto &II : Inputs) {
      if (!II.isFilename())
        continue;
      NvptxInputs.push_back(II);
    }

    constructLLVMLinkCommand(C, JA, Output, Args, SubArchName, Prefix,
                             NvptxInputs);
    return;
  }

  InputInfoList SpirvInputs;
  for (const auto &II : Inputs) {
    if (!II.isFilename())
      continue;
    SpirvInputs.push_back(II);
  }

  constructLLVMLinkCommand(C, JA, Output, Args, SubArchName, Prefix,
                           SpirvInputs);
}

static const char *makeExeName(Compilation &C, StringRef Name) {
  llvm::SmallString<8> ExeName(Name);
  const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
  if (HostTC->getTriple().isWindowsMSVCEnvironment())
    ExeName.append(".exe");
  return C.getArgs().MakeArgString(ExeName);
}

void SYCL::fpga::BackendCompiler::constructOpenCLAOTCommand(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args) const {
  // Construct opencl-aot command. This is used for FPGA AOT compilations
  // when performing emulation.  Input file will be a SPIR-V binary which
  // will be compiled to an aocx file.
  InputInfoList ForeachInputs;
  InputInfoList FPGADepFiles;
  ArgStringList CmdArgs{"-device=fpga_fast_emu"};

  for (const auto &II : Inputs) {
    if (II.getType() == types::TY_TempAOCOfilelist ||
        II.getType() == types::TY_FPGA_Dependencies ||
        II.getType() == types::TY_FPGA_Dependencies_List)
      continue;
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    CmdArgs.push_back(
        C.getArgs().MakeArgString("-spv=" + Twine(II.getFilename())));
  }
  CmdArgs.push_back(
      C.getArgs().MakeArgString("-ir=" + Twine(Output.getFilename())));

  StringRef ForeachExt = "aocx";
  if (Arg *A = Args.getLastArg(options::OPT_fsycl_link_EQ))
    if (A->getValue() == StringRef("early"))
      ForeachExt = "aocr";

  // Add any implied arguments before user defined arguments.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
  llvm::Triple CPUTriple("spir64_x86_64");
  TC.AddImpliedTargetArgs(CPUTriple, Args, CmdArgs, JA, *HostTC);
  // Add the target args passed in
  TC.TranslateBackendTargetArgs(CPUTriple, Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(CPUTriple, Args, CmdArgs);

  SmallString<128> ExecPath(
      getToolChain().GetProgramPath(makeExeName(C, "opencl-aot")));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       Exec, CmdArgs, std::nullopt);
  if (!ForeachInputs.empty()) {
    StringRef ParallelJobs =
        Args.getLastArgValue(options::OPT_fsycl_max_parallel_jobs_EQ);
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, "", ForeachExt, ParallelJobs);
  } else
    C.addCommand(std::move(Cmd));
}

void SYCL::fpga::BackendCompiler::ConstructJob(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args,
    const char *LinkingOutput) const {
  assert((getToolChain().getTriple().getArch() == llvm::Triple::spir ||
          getToolChain().getTriple().getArch() == llvm::Triple::spir64) &&
         "Unsupported target");

  // Grab the -Xsycl-target* options.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  ArgStringList TargetArgs;
  TC.TranslateBackendTargetArgs(TC.getTriple(), Args, TargetArgs);

  // When performing emulation compilations for FPGA AOT, we want to use
  // opencl-aot instead of aoc.
  if (C.getDriver().IsFPGAEmulationMode()) {
    constructOpenCLAOTCommand(C, JA, Output, Inputs, Args);
    return;
  }

  InputInfoList ForeachInputs;
  InputInfoList FPGADepFiles;
  StringRef CreatedReportName;
  ArgStringList CmdArgs{"-o", Output.getFilename()};
  for (const auto &II : Inputs) {
    std::string Filename(II.getFilename());
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    if (II.getType() == types::TY_TempAOCOfilelist)
      // Add any FPGA library lists.  These come in as special tempfile lists.
      CmdArgs.push_back(Args.MakeArgString(Twine("-library-list=") + Filename));
    else if (II.getType() == types::TY_FPGA_Dependencies ||
             II.getType() == types::TY_FPGA_Dependencies_List)
      FPGADepFiles.push_back(II);
    else
      CmdArgs.push_back(C.getArgs().MakeArgString(Filename));
    // Check for any AOCR input, if found use that as the project report name
    StringRef Ext(llvm::sys::path::extension(Filename));
    if (Ext.empty())
      continue;
    if (getToolChain().LookupTypeForExtension(Ext.drop_front()) ==
        types::TY_FPGA_AOCR) {
      // Keep the base of the .aocr file name.  Input file is a temporary,
      // so we are stripping off the additional naming information for a
      // cleaner name.  The suffix being stripped from the name is the
      // added temporary string and the extension.
      StringRef SuffixFormat("-XXXXXX.aocr");
      SmallString<128> NameBase(
          Filename.substr(0, Filename.length() - SuffixFormat.size()));
      NameBase.append(".prj");
      CreatedReportName =
          Args.MakeArgString(llvm::sys::path::filename(NameBase));
    }
  }
  CmdArgs.push_back("-sycl");

  StringRef ForeachExt = "aocx";
  if (Arg *A = Args.getLastArg(options::OPT_fsycl_link_EQ))
    if (A->getValue() == StringRef("early")) {
      CmdArgs.push_back("-rtl");
      ForeachExt = "aocr";
    }

  for (auto *A : Args) {
    // Any input file is assumed to have a dependency file associated and
    // the report folder can also be named based on the first input.
    if (A->getOption().getKind() != Option::InputClass)
      continue;
    SmallString<128> ArgName(A->getSpelling());
    StringRef Ext(llvm::sys::path::extension(ArgName));
    if (Ext.empty())
      continue;
    types::ID Ty = getToolChain().LookupTypeForExtension(Ext.drop_front());
    if (Ty == types::TY_INVALID)
      continue;
    if (types::isSrcFile(Ty) || Ty == types::TY_Object) {
      // The project report is created in CWD, so strip off any directory
      // information if provided with the input file.
      StringRef TrimmedArgName = llvm::sys::path::filename(ArgName);
      if (types::isSrcFile(Ty)) {
        SmallString<128> DepName(
            C.getDriver().getFPGATempDepFile(std::string(TrimmedArgName)));
        if (!DepName.empty())
          FPGADepFiles.push_back(InputInfo(types::TY_Dependencies,
                                           Args.MakeArgString(DepName),
                                           Args.MakeArgString(DepName)));
      }
      if (CreatedReportName.empty()) {
        // Project report should be saved into CWD, so strip off any
        // directory information if provided with the input file.
        llvm::sys::path::replace_extension(ArgName, "prj");
        CreatedReportName = Args.MakeArgString(ArgName);
      }
    }
  }

  // Add any dependency files.
  if (!FPGADepFiles.empty()) {
    SmallString<128> DepOpt("-dep-files=");
    for (unsigned I = 0; I < FPGADepFiles.size(); ++I) {
      if (I)
        DepOpt += ',';
      if (FPGADepFiles[I].getType() == types::TY_FPGA_Dependencies_List)
        DepOpt += "@";
      DepOpt += FPGADepFiles[I].getFilename();
    }
    CmdArgs.push_back(C.getArgs().MakeArgString(DepOpt));
  }

  // Depending on output file designations, set the report folder
  SmallString<128> ReportOptArg;
  if (Arg *FinalOutput = Args.getLastArg(options::OPT_o, options::OPT__SLASH_o,
                                         options::OPT__SLASH_Fe)) {
    SmallString<128> FN(FinalOutput->getValue());
    // For "-o file.xxx" where the option value has an extension, if the
    // extension is one of .a .o .out .lib .obj .exe, the output project
    // directory name will be file.proj which omits the extension. Otherwise
    // the output project directory name will be file.xxx.prj which keeps
    // the original extension.
    StringRef Ext = llvm::sys::path::extension(FN);
    SmallVector<StringRef, 6> Exts = {".o",   ".a",   ".out",
                                      ".obj", ".lib", ".exe"};
    if (std::find(Exts.begin(), Exts.end(), Ext) != Exts.end())
      llvm::sys::path::replace_extension(FN, "prj");
    else
      FN.append(".prj");
    const char *FolderName = Args.MakeArgString(FN);
    ReportOptArg += FolderName;
  } else {
    // Default output directory should match default output executable name
    ReportOptArg += "a.prj";
  }
  if (!ReportOptArg.empty())
    CmdArgs.push_back(C.getArgs().MakeArgString(
        Twine("-output-report-folder=") + ReportOptArg));

  // Add any implied arguments before user defined arguments.
  const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
  TC.AddImpliedTargetArgs(getToolChain().getTriple(), Args, CmdArgs, JA,
                          *HostTC);

  // Add -Xsycl-target* options.
  TC.TranslateBackendTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(getToolChain().getTriple(), Args, CmdArgs);

  // Look for -reuse-exe=XX option
  if (Arg *A = Args.getLastArg(options::OPT_reuse_exe_EQ)) {
    Args.ClaimAllArgs(options::OPT_reuse_exe_EQ);
    CmdArgs.push_back(Args.MakeArgString(A->getAsString(Args)));
  }

  SmallString<128> ExecPath(
      getToolChain().GetProgramPath(makeExeName(C, "aoc")));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       Exec, CmdArgs, std::nullopt);
  addFPGATimingDiagnostic(Cmd, C);
  if (!ForeachInputs.empty()) {
    StringRef ParallelJobs =
        Args.getLastArgValue(options::OPT_fsycl_max_parallel_jobs_EQ);
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, ReportOptArg, ForeachExt, ParallelJobs);
  } else
    C.addCommand(std::move(Cmd));
}

static llvm::StringMap<StringRef> GRFModeFlagMap{
    {"auto", "-ze-intel-enable-auto-large-GRF-mode"},
    {"small", "-ze-intel-128-GRF-per-thread"},
    {"large", "-ze-opt-large-register-file"}};

StringRef SYCL::gen::getGenGRFFlag(StringRef GRFMode) {
  if (!GRFModeFlagMap.contains(GRFMode))
    return "";
  return GRFModeFlagMap[GRFMode];
}

void SYCL::gen::BackendCompiler::ConstructJob(Compilation &C,
                                              const JobAction &JA,
                                              const InputInfo &Output,
                                              const InputInfoList &Inputs,
                                              const ArgList &Args,
                                              const char *LinkingOutput) const {
  assert((getToolChain().getTriple().getArch() == llvm::Triple::spir ||
          getToolChain().getTriple().getArch() == llvm::Triple::spir64) &&
         "Unsupported target");
  ArgStringList CmdArgs{"-output", Output.getFilename()};
  InputInfoList ForeachInputs;
  for (const auto &II : Inputs) {
    CmdArgs.push_back("-file");
    std::string Filename(II.getFilename());
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    CmdArgs.push_back(C.getArgs().MakeArgString(Filename));
  }
  // The next line prevents ocloc from modifying the image name
  CmdArgs.push_back("-output_no_suffix");
  CmdArgs.push_back("-spirv_input");
  StringRef Device = JA.getOffloadingArch();

  // Add -Xsycl-target* options.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
  TC.AddImpliedTargetArgs(getToolChain().getTriple(), Args, CmdArgs, JA,
                          *HostTC);
  TC.TranslateBackendTargetArgs(getToolChain().getTriple(), Args, CmdArgs,
                                Device);
  TC.TranslateLinkerTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  SmallString<128> ExecPath(
      getToolChain().GetProgramPath(makeExeName(C, "ocloc")));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       Exec, CmdArgs, std::nullopt);
  if (!ForeachInputs.empty()) {
    StringRef ParallelJobs =
        Args.getLastArgValue(options::OPT_fsycl_max_parallel_jobs_EQ);
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, "", "out", ParallelJobs);
  } else
    C.addCommand(std::move(Cmd));
}

StringRef SYCL::gen::resolveGenDevice(StringRef DeviceName) {
  StringRef Device;
  Device =
      llvm::StringSwitch<StringRef>(DeviceName)
          .Cases("intel_gpu_bdw", "intel_gpu_8_0_0", "bdw")
          .Cases("intel_gpu_skl", "intel_gpu_9_0_9", "skl")
          .Cases("intel_gpu_kbl", "intel_gpu_9_1_9", "kbl")
          .Cases("intel_gpu_cfl", "intel_gpu_9_2_9", "cfl")
          .Cases("intel_gpu_apl", "intel_gpu_bxt", "intel_gpu_9_3_0", "apl")
          .Cases("intel_gpu_glk", "intel_gpu_9_4_0", "glk")
          .Cases("intel_gpu_whl", "intel_gpu_9_5_0", "whl")
          .Cases("intel_gpu_aml", "intel_gpu_9_6_0", "aml")
          .Cases("intel_gpu_cml", "intel_gpu_9_7_0", "cml")
          .Cases("intel_gpu_icllp", "intel_gpu_11_0_0", "icllp")
          .Cases("intel_gpu_ehl", "intel_gpu_jsl", "ehl")
          .Cases("intel_gpu_tgllp", "intel_gpu_12_0_0", "tgllp")
          .Case("intel_gpu_rkl", "rkl")
          .Cases("intel_gpu_adl_s", "intel_gpu_rpl_s", "adl_s")
          .Case("intel_gpu_adl_p", "adl_p")
          .Case("intel_gpu_adl_n", "adl_n")
          .Cases("intel_gpu_dg1", "intel_gpu_12_10_0", "dg1")
          .Cases("intel_gpu_acm_g10", "intel_gpu_dg2_g10", "acm_g10")
          .Cases("intel_gpu_acm_g11", "intel_gpu_dg2_g11", "acm_g11")
          .Cases("intel_gpu_acm_g12", "intel_gpu_dg2_g12", "acm_g12")
          .Case("intel_gpu_pvc", "pvc")
          .Case("intel_gpu_pvc_vg", "pvc_vg")
          .Case("nvidia_gpu_sm_50", "sm_50")
          .Case("nvidia_gpu_sm_52", "sm_52")
          .Case("nvidia_gpu_sm_53", "sm_53")
          .Case("nvidia_gpu_sm_60", "sm_60")
          .Case("nvidia_gpu_sm_61", "sm_61")
          .Case("nvidia_gpu_sm_62", "sm_62")
          .Case("nvidia_gpu_sm_70", "sm_70")
          .Case("nvidia_gpu_sm_72", "sm_72")
          .Case("nvidia_gpu_sm_75", "sm_75")
          .Case("nvidia_gpu_sm_80", "sm_80")
          .Case("nvidia_gpu_sm_86", "sm_86")
          .Case("nvidia_gpu_sm_87", "sm_87")
          .Case("nvidia_gpu_sm_89", "sm_89")
          .Case("nvidia_gpu_sm_90", "sm_90")
          .Case("amd_gpu_gfx700", "gfx700")
          .Case("amd_gpu_gfx701", "gfx701")
          .Case("amd_gpu_gfx702", "gfx702")
          .Case("amd_gpu_gfx801", "gfx801")
          .Case("amd_gpu_gfx802", "gfx802")
          .Case("amd_gpu_gfx803", "gfx803")
          .Case("amd_gpu_gfx805", "gfx805")
          .Case("amd_gpu_gfx810", "gfx810")
          .Case("amd_gpu_gfx900", "gfx900")
          .Case("amd_gpu_gfx902", "gfx902")
          .Case("amd_gpu_gfx904", "gfx904")
          .Case("amd_gpu_gfx906", "gfx906")
          .Case("amd_gpu_gfx908", "gfx908")
          .Case("amd_gpu_gfx909", "gfx909")
          .Case("amd_gpu_gfx90a", "gfx90a")
          .Case("amd_gpu_gfx90c", "gfx90c")
          .Case("amd_gpu_gfx940", "gfx940")
          .Case("amd_gpu_gfx941", "gfx941")
          .Case("amd_gpu_gfx942", "gfx942")
          .Case("amd_gpu_gfx1010", "gfx1010")
          .Case("amd_gpu_gfx1011", "gfx1011")
          .Case("amd_gpu_gfx1012", "gfx1012")
          .Case("amd_gpu_gfx1013", "gfx1013")
          .Case("amd_gpu_gfx1030", "gfx1030")
          .Case("amd_gpu_gfx1031", "gfx1031")
          .Case("amd_gpu_gfx1032", "gfx1032")
          .Case("amd_gpu_gfx1033", "gfx1033")
          .Case("amd_gpu_gfx1034", "gfx1034")
          .Case("amd_gpu_gfx1035", "gfx1035")
          .Case("amd_gpu_gfx1036", "gfx1036")
          .Case("amd_gpu_gfx1100", "gfx1100")
          .Case("amd_gpu_gfx1101", "gfx1101")
          .Case("amd_gpu_gfx1102", "gfx1102")
          .Case("amd_gpu_gfx1103", "gfx1103")
          .Case("amd_gpu_gfx1150", "gfx1150")
          .Case("amd_gpu_gfx1151", "gfx1151")
          .Case("amd_gpu_gfx1200", "gfx1200")
          .Case("amd_gpu_gfx1201", "gfx1201")
          .Default("");
  return Device;
}

SmallString<64> SYCL::gen::getGenDeviceMacro(StringRef DeviceName) {
  SmallString<64> Macro;
  StringRef Ext = llvm::StringSwitch<StringRef>(DeviceName)
                      .Case("bdw", "INTEL_GPU_BDW")
                      .Case("skl", "INTEL_GPU_SKL")
                      .Case("kbl", "INTEL_GPU_KBL")
                      .Case("cfl", "INTEL_GPU_CFL")
                      .Case("apl", "INTEL_GPU_APL")
                      .Case("glk", "INTEL_GPU_GLK")
                      .Case("whl", "INTEL_GPU_WHL")
                      .Case("aml", "INTEL_GPU_AML")
                      .Case("cml", "INTEL_GPU_CML")
                      .Case("icllp", "INTEL_GPU_ICLLP")
                      .Case("ehl", "INTEL_GPU_EHL")
                      .Case("tgllp", "INTEL_GPU_TGLLP")
                      .Case("rkl", "INTEL_GPU_RKL")
                      .Case("adl_s", "INTEL_GPU_ADL_S")
                      .Case("adl_p", "INTEL_GPU_ADL_P")
                      .Case("adl_n", "INTEL_GPU_ADL_N")
                      .Case("dg1", "INTEL_GPU_DG1")
                      .Case("acm_g10", "INTEL_GPU_ACM_G10")
                      .Case("acm_g11", "INTEL_GPU_ACM_G11")
                      .Case("acm_g12", "INTEL_GPU_ACM_G12")
                      .Case("pvc", "INTEL_GPU_PVC")
                      .Case("pvc_vg", "INTEL_GPU_PVC_VG")
                      .Case("sm_50", "NVIDIA_GPU_SM_50")
                      .Case("sm_52", "NVIDIA_GPU_SM_52")
                      .Case("sm_53", "NVIDIA_GPU_SM_53")
                      .Case("sm_60", "NVIDIA_GPU_SM_60")
                      .Case("sm_61", "NVIDIA_GPU_SM_61")
                      .Case("sm_62", "NVIDIA_GPU_SM_62")
                      .Case("sm_70", "NVIDIA_GPU_SM_70")
                      .Case("sm_72", "NVIDIA_GPU_SM_72")
                      .Case("sm_75", "NVIDIA_GPU_SM_75")
                      .Case("sm_80", "NVIDIA_GPU_SM_80")
                      .Case("sm_86", "NVIDIA_GPU_SM_86")
                      .Case("sm_87", "NVIDIA_GPU_SM_87")
                      .Case("sm_89", "NVIDIA_GPU_SM_89")
                      .Case("sm_90", "NVIDIA_GPU_SM_90")
                      .Case("gfx700", "AMD_GPU_GFX700")
                      .Case("gfx701", "AMD_GPU_GFX701")
                      .Case("gfx702", "AMD_GPU_GFX702")
                      .Case("gfx801", "AMD_GPU_GFX801")
                      .Case("gfx802", "AMD_GPU_GFX802")
                      .Case("gfx803", "AMD_GPU_GFX803")
                      .Case("gfx805", "AMD_GPU_GFX805")
                      .Case("gfx810", "AMD_GPU_GFX810")
                      .Case("gfx900", "AMD_GPU_GFX900")
                      .Case("gfx902", "AMD_GPU_GFX902")
                      .Case("gfx904", "AMD_GPU_GFX904")
                      .Case("gfx906", "AMD_GPU_GFX906")
                      .Case("gfx908", "AMD_GPU_GFX908")
                      .Case("gfx909", "AMD_GPU_GFX909")
                      .Case("gfx90a", "AMD_GPU_GFX90A")
                      .Case("gfx90c", "AMD_GPU_GFX90C")
                      .Case("gfx940", "AMD_GPU_GFX940")
                      .Case("gfx941", "AMD_GPU_GFX941")
                      .Case("gfx942", "AMD_GPU_GFX942")
                      .Case("gfx1010", "AMD_GPU_GFX1010")
                      .Case("gfx1011", "AMD_GPU_GFX1011")
                      .Case("gfx1012", "AMD_GPU_GFX1012")
                      .Case("gfx1013", "AMD_GPU_GFX1013")
                      .Case("gfx1030", "AMD_GPU_GFX1030")
                      .Case("gfx1031", "AMD_GPU_GFX1031")
                      .Case("gfx1032", "AMD_GPU_GFX1032")
                      .Case("gfx1033", "AMD_GPU_GFX1033")
                      .Case("gfx1034", "AMD_GPU_GFX1034")
                      .Case("gfx1035", "AMD_GPU_GFX1035")
                      .Case("gfx1036", "AMD_GPU_GFX1036")
                      .Case("gfx1100", "AMD_GPU_GFX1100")
                      .Case("gfx1101", "AMD_GPU_GFX1101")
                      .Case("gfx1102", "AMD_GPU_GFX1102")
                      .Case("gfx1103", "AMD_GPU_GFX1103")
                      .Case("gfx1150", "AMD_GPU_GFX1150")
                      .Case("gfx1151", "AMD_GPU_GFX1151")
                      .Case("gfx1200", "AMD_GPU_GFX1200")
                      .Case("gfx1201", "AMD_GPU_GFX1201")
                      .Default("");
  if (!Ext.empty()) {
    Macro = "__SYCL_TARGET_";
    Macro += Ext;
    Macro += "__";
  }
  return Macro;
}

void SYCL::x86_64::BackendCompiler::ConstructJob(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args,
    const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  CmdArgs.push_back(Args.MakeArgString(Twine("-o=") + Output.getFilename()));
  CmdArgs.push_back("--device=cpu");
  InputInfoList ForeachInputs;
  for (const auto &II : Inputs) {
    std::string Filename(II.getFilename());
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    CmdArgs.push_back(Args.MakeArgString(Filename));
  }
  // Add -Xsycl-target* options.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
  TC.AddImpliedTargetArgs(getToolChain().getTriple(), Args, CmdArgs, JA,
                          *HostTC);
  TC.TranslateBackendTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  SmallString<128> ExecPath(
      getToolChain().GetProgramPath(makeExeName(C, "opencl-aot")));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       Exec, CmdArgs, std::nullopt);
  if (!ForeachInputs.empty()) {
    StringRef ParallelJobs =
        Args.getLastArgValue(options::OPT_fsycl_max_parallel_jobs_EQ);
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, "", "out", ParallelJobs);
  } else
    C.addCommand(std::move(Cmd));
}

// Unsupported options for device compilation
//  -fcf-protection, -fsanitize, -fprofile-generate, -fprofile-instr-generate
//  -ftest-coverage, -fcoverage-mapping, -fcreate-profile, -fprofile-arcs
//  -fcs-profile-generate -forder-file-instrumentation
static std::vector<OptSpecifier> getUnsupportedOpts(void) {
  std::vector<OptSpecifier> UnsupportedOpts = {
      options::OPT_fsanitize_EQ,
      options::OPT_fcf_protection_EQ,
      options::OPT_fprofile_generate,
      options::OPT_fprofile_generate_EQ,
      options::OPT_fno_profile_generate,
      options::OPT_ftest_coverage,
      options::OPT_fno_test_coverage,
      options::OPT_fcoverage_mapping,
      options::OPT_fno_coverage_mapping,
      options::OPT_fprofile_instr_generate,
      options::OPT_fprofile_instr_generate_EQ,
      options::OPT_fprofile_arcs,
      options::OPT_fno_profile_arcs,
      options::OPT_fno_profile_instr_generate,
      options::OPT_fcreate_profile,
      options::OPT_fprofile_instr_use,
      options::OPT_fprofile_instr_use_EQ,
      options::OPT_forder_file_instrumentation,
      options::OPT_fcs_profile_generate,
      options::OPT_fcs_profile_generate_EQ};
  return UnsupportedOpts;
}

SYCLToolChain::SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC),
      IsSYCLNativeCPU(Triple == HostTC.getTriple()) {
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);

  // Diagnose unsupported options only once.
  for (OptSpecifier Opt : getUnsupportedOpts()) {
    if (const Arg *A = Args.getLastArg(Opt)) {
      // All sanitizer options are not currently supported, except
      // AddressSanitizer
      if (A->getOption().getID() == options::OPT_fsanitize_EQ &&
          A->getValues().size() == 1) {
        std::string SanitizeVal = A->getValue();
        if (SanitizeVal == "address")
          continue;
      }
      D.Diag(clang::diag::warn_drv_unsupported_option_for_target)
          << A->getAsString(Args) << getTriple().str();
    }
  }
}

void SYCLToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);
}

llvm::opt::DerivedArgList *
SYCLToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);

  bool IsNewDAL = false;
  if (!DAL) {
    DAL = new DerivedArgList(Args.getBaseArgs());
    IsNewDAL = true;
  }

  for (Arg *A : Args) {
    // Filter out any options we do not want to pass along to the device
    // compilation.
    auto Opt(A->getOption());
    bool Unsupported = false;
    for (OptSpecifier UnsupportedOpt : getUnsupportedOpts()) {
      if (Opt.matches(UnsupportedOpt)) {
        if (Opt.getID() == options::OPT_fsanitize_EQ &&
            A->getValues().size() == 1) {
          std::string SanitizeVal = A->getValue();
          if (SanitizeVal == "address") {
            if (IsNewDAL)
              DAL->append(A);
            continue;
          }
        }
        if (!IsNewDAL)
          DAL->eraseArg(Opt.getID());
        Unsupported = true;
      }
    }
    if (Unsupported)
      continue;
    if (IsNewDAL)
      DAL->append(A);
  }
  // Strip out -O0 for FPGA Hardware device compilation.
  if (getDriver().IsFPGAHWMode() &&
      getTriple().getSubArch() == llvm::Triple::SPIRSubArch_fpga)
    DAL->eraseArg(options::OPT_O0);

  const OptTable &Opts = getDriver().getOpts();
  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                      BoundArch);
  }
  return DAL;
}

static void parseTargetOpts(StringRef ArgString, const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs) {
  // Tokenize the string.
  SmallVector<const char *, 8> TargetArgs;
  llvm::BumpPtrAllocator A;
  llvm::StringSaver S(A);
  llvm::cl::TokenizeGNUCommandLine(ArgString, S, TargetArgs);
  for (StringRef TA : TargetArgs)
    CmdArgs.push_back(Args.MakeArgString(TA));
}

void SYCLToolChain::TranslateGPUTargetOpt(const llvm::opt::ArgList &Args,
                                          llvm::opt::ArgStringList &CmdArgs,
                                          OptSpecifier Opt_EQ) const {
  for (auto *A : Args) {
    if (A->getOption().matches(Opt_EQ)) {
      if (auto GpuDevice =
              tools::SYCL::gen::isGPUTarget<tools::SYCL::gen::AmdGPU>(
                  A->getValue())) {
        StringRef ArgString;
        SmallString<64> OffloadArch("--offload-arch=");
        OffloadArch += GpuDevice->data();
        ArgString = OffloadArch;
        parseTargetOpts(ArgString, Args, CmdArgs);
        A->claim();
      }
    }
  }
}

static void WarnForDeprecatedBackendOpts(const Driver &D,
                                         const llvm::Triple &Triple,
                                         StringRef Device, StringRef ArgString,
                                         const llvm::opt::Arg *A) {
  // Suggest users passing GRF backend opts on PVC to use
  // -ftarget-register-alloc-mode and

  if (!ArgString.contains("-device pvc") && !Device.contains("pvc"))
    return;
  // Make sure to only warn for once for gen targets as the translate
  // options tree is called twice but only the second time has the
  // device set.
  if (Triple.isSPIR() && Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen &&
      !A->isClaimed())
    return;
  for (const auto &[Mode, Flag] : GRFModeFlagMap)
    if (ArgString.contains(Flag))
      D.Diag(diag::warn_drv_ftarget_register_alloc_mode_pvc) << Flag << Mode;
}

// Expects a specific type of option (e.g. -Xsycl-target-backend) and will
// extract the arguments.
void SYCLToolChain::TranslateTargetOpt(const llvm::opt::ArgList &Args,
                                       llvm::opt::ArgStringList &CmdArgs,
                                       OptSpecifier Opt, OptSpecifier Opt_EQ,
                                       StringRef Device) const {
  for (auto *A : Args) {
    bool OptNoTriple;
    OptNoTriple = A->getOption().matches(Opt);
    if (A->getOption().matches(Opt_EQ)) {
      // Passing device args: -X<Opt>=<triple> -opt=val.
      StringRef GenDevice = SYCL::gen::resolveGenDevice(A->getValue());
      bool IsGenTriple =
          getTriple().isSPIR() &&
          getTriple().getSubArch() == llvm::Triple::SPIRSubArch_gen;
      if (Device != GenDevice)
        continue;
      if (getDriver().MakeSYCLDeviceTriple(A->getValue()) != getTriple() &&
          (!IsGenTriple || (IsGenTriple && GenDevice.empty())))
        // Triples do not match, but only skip when we know we are not comparing
        // against intel_gpu_* and non-spir64_gen
        continue;
    } else if (!OptNoTriple)
      // Don't worry about any of the other args, we only want to pass what is
      // passed in -X<Opt>
      continue;

    // Add the argument from -X<Opt>
    StringRef ArgString;
    if (OptNoTriple) {
      // With multiple -fsycl-targets, a triple is required so we know where
      // the options should go.
      const Arg *TargetArg = Args.getLastArg(options::OPT_fsycl_targets_EQ);
      if (TargetArg && TargetArg->getValues().size() != 1) {
        getDriver().Diag(diag::err_drv_Xsycl_target_missing_triple)
            << A->getSpelling();
        continue;
      }
      // No triple, so just add the argument.
      ArgString = A->getValue();
    } else
      // Triple found, add the next argument in line.
      ArgString = A->getValue(1);
    WarnForDeprecatedBackendOpts(getDriver(), getTriple(), Device, ArgString,
                                 A);
    parseTargetOpts(ArgString, Args, CmdArgs);
    A->claim();
  }
}

void SYCLToolChain::AddImpliedTargetArgs(const llvm::Triple &Triple,
                                         const llvm::opt::ArgList &Args,
                                         llvm::opt::ArgStringList &CmdArgs,
                                         const JobAction &JA,
                                         const ToolChain &HostTC) const {
  // Current implied args are for debug information and disabling of
  // optimizations.  They are passed along to the respective areas as follows:
  // FPGA:  -g -cl-opt-disable
  // Default device AOT: -g -cl-opt-disable
  // Default device JIT: -g (-O0 is handled by the runtime)
  // GEN:  -options "-g -O0"
  // CPU:  "--bo=-g -cl-opt-disable"
  llvm::opt::ArgStringList BeArgs;
  // Per-device argument vector storing the device name and the backend argument
  // string
  llvm::SmallVector<std::pair<StringRef, StringRef>, 16> PerDeviceArgs;
  bool IsGen = Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen;
  if (Arg *A = Args.getLastArg(options::OPT_g_Group, options::OPT__SLASH_Z7))
    if (!A->getOption().matches(options::OPT_g0))
      BeArgs.push_back("-g");
  // Only pass -cl-opt-disable for non-JIT, as the runtime
  // handles O0 for the JIT case.
  if (Triple.getSubArch() != llvm::Triple::NoSubArch)
    if (Arg *A = Args.getLastArg(options::OPT_O_Group))
      if (A->getOption().matches(options::OPT_O0))
        BeArgs.push_back("-cl-opt-disable");
  StringRef RegAllocModeOptName = "-ftarget-register-alloc-mode=";
  if (Arg *A = Args.getLastArg(options::OPT_ftarget_register_alloc_mode_EQ)) {
    StringRef RegAllocModeVal = A->getValue(0);
    auto ProcessElement = [&](StringRef Ele) {
      auto [DeviceName, RegAllocMode] = Ele.split(':');
      StringRef BackendOptName = SYCL::gen::getGenGRFFlag(RegAllocMode);
      bool IsDefault = RegAllocMode.equals("default");
      if (RegAllocMode.empty() || !DeviceName.equals("pvc") ||
          (BackendOptName.empty() && !IsDefault)) {
        getDriver().Diag(diag::err_drv_unsupported_option_argument)
            << A->getSpelling() << Ele;
      }
      // "default" means "provide no specification to the backend", so
      // we don't need to do anything here.
      if (IsDefault)
        return;
      if (IsGen) {
        // For AOT, Use ocloc's per-device options flag with the correct ocloc
        // option to honor the user's specification.
        PerDeviceArgs.push_back(
            {DeviceName, Args.MakeArgString(BackendOptName)});
      } else if (Triple.isSPIR() &&
                 Triple.getSubArch() == llvm::Triple::NoSubArch) {
        // For JIT, pass -ftarget-register-alloc-mode=Device:BackendOpt to
        // clang-offload-wrapper to be processed by the runtime.
        BeArgs.push_back(Args.MakeArgString(RegAllocModeOptName + DeviceName +
                                            ":" + BackendOptName));
      }
    };
    llvm::SmallVector<StringRef, 16> RegAllocModeArgs;
    RegAllocModeVal.split(RegAllocModeArgs, ',');
    for (StringRef Elem : RegAllocModeArgs)
      ProcessElement(Elem);
  } else if (!HostTC.getTriple().isWindowsMSVCEnvironment()) {
    // If -ftarget-register-alloc-mode is not specified, the default is
    // pvc:default on Windows and and pvc:auto otherwise.
    StringRef DeviceName = "pvc";
    StringRef BackendOptName = SYCL::gen::getGenGRFFlag("auto");
    if (IsGen)
      PerDeviceArgs.push_back({DeviceName, Args.MakeArgString(BackendOptName)});
    else if (Triple.isSPIR() &&
             Triple.getSubArch() == llvm::Triple::NoSubArch) {
      BeArgs.push_back(Args.MakeArgString(RegAllocModeOptName + DeviceName +
                                          ":" + BackendOptName));
    }
  }
  if (IsGen) {
    // For GEN (spir64_gen) we have implied -device settings given usage
    // of intel_gpu_ as a target.  Handle those here, and also check that no
    // other -device was passed, as that is a conflict.
    StringRef DepInfo = JA.getOffloadingArch();
    if (!DepInfo.empty()) {
      ArgStringList TargArgs;
      Args.AddAllArgValues(TargArgs, options::OPT_Xs, options::OPT_Xs_separate);
      Args.AddAllArgValues(TargArgs, options::OPT_Xsycl_backend);
      // For -Xsycl-target-backend=<triple> we need to scrutinize the triple
      for (auto *A : Args) {
        if (!A->getOption().matches(options::OPT_Xsycl_backend_EQ))
          continue;
        if (StringRef(A->getValue()).starts_with("intel_gpu"))
          TargArgs.push_back(A->getValue(1));
      }
      if (llvm::find_if(TargArgs, [&](auto Cur) {
            return !strncmp(Cur, "-device", sizeof("-device") - 1);
          }) != TargArgs.end()) {
        SmallString<64> Target("intel_gpu_");
        Target += DepInfo;
        getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
            << "-device" << Target;
      }
      CmdArgs.push_back("-device");
      CmdArgs.push_back(Args.MakeArgString(DepInfo));
    }
    // -ftarget-compile-fast AOT
    if (Args.hasArg(options::OPT_ftarget_compile_fast))
      BeArgs.push_back("-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'");
    // -ftarget-export-symbols
    if (Args.hasFlag(options::OPT_ftarget_export_symbols,
                     options::OPT_fno_target_export_symbols, false))
      BeArgs.push_back("-library-compilation");
  } else if (Triple.getSubArch() == llvm::Triple::NoSubArch &&
             Triple.isSPIR()) {
    // -ftarget-compile-fast JIT
    Args.AddLastArg(BeArgs, options::OPT_ftarget_compile_fast);
  }
  if (IsGen) {
    for (auto [DeviceName, BackendArgStr] : PerDeviceArgs) {
      CmdArgs.push_back("-device_options");
      CmdArgs.push_back(Args.MakeArgString(DeviceName));
      CmdArgs.push_back(Args.MakeArgString(BackendArgStr));
    }
  }
  if (BeArgs.empty())
    return;
  if (Triple.getSubArch() == llvm::Triple::NoSubArch ||
      Triple.getSubArch() == llvm::Triple::SPIRSubArch_fpga) {
    for (StringRef A : BeArgs)
      CmdArgs.push_back(Args.MakeArgString(A));
    return;
  }
  SmallString<128> BeOpt;
  if (IsGen)
    CmdArgs.push_back("-options");
  else
    BeOpt = "--bo=";
  for (unsigned I = 0; I < BeArgs.size(); ++I) {
    if (I)
      BeOpt += ' ';
    BeOpt += BeArgs[I];
  }
  CmdArgs.push_back(Args.MakeArgString(BeOpt));
}

void SYCLToolChain::TranslateBackendTargetArgs(
    const llvm::Triple &Triple, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CmdArgs, StringRef Device) const {
  // Handle -Xs flags.
  for (auto *A : Args) {
    // When parsing the target args, the -Xs<opt> type option applies to all
    // target compilations is not associated with a specific triple.  The
    // option can be used in 3 different ways:
    //   -Xs -DFOO -Xs -DBAR
    //   -Xs "-DFOO -DBAR"
    //   -XsDFOO -XsDBAR
    // All of the above examples will pass -DFOO -DBAR to the backend compiler.

    // Do not add the -Xs to the default SYCL triple (spir64) when we know we
    // have implied the setting.
    if ((A->getOption().matches(options::OPT_Xs) ||
         A->getOption().matches(options::OPT_Xs_separate)) &&
        Triple.getSubArch() == llvm::Triple::NoSubArch && Triple.isSPIR() &&
        getDriver().isSYCLDefaultTripleImplied())
      continue;

    if (A->getOption().matches(options::OPT_Xs)) {
      // Take the arg and create an option out of it.
      CmdArgs.push_back(Args.MakeArgString(Twine("-") + A->getValue()));
      WarnForDeprecatedBackendOpts(getDriver(), Triple, Device, A->getValue(),
                                   A);
      A->claim();
      continue;
    }
    if (A->getOption().matches(options::OPT_Xs_separate)) {
      StringRef ArgString(A->getValue());
      parseTargetOpts(ArgString, Args, CmdArgs);
      WarnForDeprecatedBackendOpts(getDriver(), Triple, Device, ArgString, A);
      A->claim();
      continue;
    }
  }
  // Do not process -Xsycl-target-backend for implied spir64
  if (Triple.getSubArch() == llvm::Triple::NoSubArch && Triple.isSPIR() &&
      getDriver().isSYCLDefaultTripleImplied())
    return;
  // Handle -Xsycl-target-backend.
  TranslateTargetOpt(Args, CmdArgs, options::OPT_Xsycl_backend,
                     options::OPT_Xsycl_backend_EQ, Device);
  TranslateGPUTargetOpt(Args, CmdArgs, options::OPT_fsycl_targets_EQ);
}

void SYCLToolChain::TranslateLinkerTargetArgs(
    const llvm::Triple &Triple, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CmdArgs) const {
  // Do not process -Xsycl-target-linker for implied spir64
  if (Triple.getSubArch() == llvm::Triple::NoSubArch && Triple.isSPIR() &&
      getDriver().isSYCLDefaultTripleImplied())
    return;
  // Handle -Xsycl-target-linker.
  TranslateTargetOpt(Args, CmdArgs, options::OPT_Xsycl_linker,
                     options::OPT_Xsycl_linker_EQ, StringRef());
}

Tool *SYCLToolChain::buildBackendCompiler() const {
  if (getTriple().getSubArch() == llvm::Triple::SPIRSubArch_fpga)
    return new tools::SYCL::fpga::BackendCompiler(*this);
  if (getTriple().getSubArch() == llvm::Triple::SPIRSubArch_gen)
    return new tools::SYCL::gen::BackendCompiler(*this);
  // fall through is CPU.
  return new tools::SYCL::x86_64::BackendCompiler(*this);
}

Tool *SYCLToolChain::buildLinker() const {
  assert(getTriple().getArch() == llvm::Triple::spir ||
         getTriple().getArch() == llvm::Triple::spir64 || IsSYCLNativeCPU);
  return new tools::SYCL::Linker(*this);
}

void SYCLToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
SYCLToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void SYCLToolChain::AddSYCLIncludeArgs(const clang::driver::Driver &Driver,
                                       const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) {
  // Add ../include/sycl, ../include/sycl/stl_wrappers and ../include (in that
  // order).
  SmallString<128> IncludePath(Driver.getInstalledDir());
  llvm::sys::path::append(IncludePath, "..");
  llvm::sys::path::append(IncludePath, "include");
  SmallString<128> SYCLPath(IncludePath);
  llvm::sys::path::append(SYCLPath, "sycl");
  // This is used to provide our wrappers around STL headers that provide
  // additional functions/template specializations when the user includes those
  // STL headers in their programs (e.g., <complex>).
  SmallString<128> STLWrappersPath(SYCLPath);
  llvm::sys::path::append(STLWrappersPath, "stl_wrappers");
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(SYCLPath));
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(STLWrappersPath));
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(IncludePath));
}

void SYCLToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

SanitizerMask SYCLToolChain::getSupportedSanitizers() const {
  return SanitizerKind::Address;
}
