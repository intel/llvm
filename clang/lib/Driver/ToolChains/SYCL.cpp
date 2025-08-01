//===--- SYCL.cpp - SYCL Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SYCL.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/SYCLLowerIR/DeviceConfigFile.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include <sstream>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

// Struct that relates an AOT target value with
// Intel CPUs and Intel GPUs.
struct StringToOffloadArchSYCLMap {
  const char *ArchName;
  SYCLSupportedIntelArchs IntelArch;
};

// Mapping of supported SYCL offloading architectures.
static const StringToOffloadArchSYCLMap StringToArchNamesMap[] = {
    // Intel CPU mapping.
    {"skylake-avx512", SYCLSupportedIntelArchs::SKYLAKEAVX512},
    {"core-avx2", SYCLSupportedIntelArchs::COREAVX2},
    {"corei7-avx", SYCLSupportedIntelArchs::COREI7AVX},
    {"corei7", SYCLSupportedIntelArchs::COREI7},
    {"westmere", SYCLSupportedIntelArchs::WESTMERE},
    {"sandybridge", SYCLSupportedIntelArchs::SANDYBRIDGE},
    {"ivybridge", SYCLSupportedIntelArchs::IVYBRIDGE},
    {"broadwell", SYCLSupportedIntelArchs::BROADWELL},
    {"coffeelake", SYCLSupportedIntelArchs::COFFEELAKE},
    {"alderlake", SYCLSupportedIntelArchs::ALDERLAKE},
    {"skylake", SYCLSupportedIntelArchs::SKYLAKE},
    {"skx", SYCLSupportedIntelArchs::SKX},
    {"cascadelake", SYCLSupportedIntelArchs::CASCADELAKE},
    {"icelake-client", SYCLSupportedIntelArchs::ICELAKECLIENT},
    {"icelake-server", SYCLSupportedIntelArchs::ICELAKESERVER},
    {"sapphirerapids", SYCLSupportedIntelArchs::SAPPHIRERAPIDS},
    {"graniterapids", SYCLSupportedIntelArchs::GRANITERAPIDS},
    // Intel GPU mapping.
    {"bdw", SYCLSupportedIntelArchs::BDW},
    {"skl", SYCLSupportedIntelArchs::SKL},
    {"kbl", SYCLSupportedIntelArchs::KBL},
    {"cfl", SYCLSupportedIntelArchs::CFL},
    {"apl", SYCLSupportedIntelArchs::APL},
    {"bxt", SYCLSupportedIntelArchs::BXT},
    {"glk", SYCLSupportedIntelArchs::GLK},
    {"whl", SYCLSupportedIntelArchs::WHL},
    {"aml", SYCLSupportedIntelArchs::AML},
    {"cml", SYCLSupportedIntelArchs::CML},
    {"icllp", SYCLSupportedIntelArchs::ICLLP},
    {"icl", SYCLSupportedIntelArchs::ICL},
    {"ehl", SYCLSupportedIntelArchs::EHL},
    {"jsl", SYCLSupportedIntelArchs::JSL},
    {"tgllp", SYCLSupportedIntelArchs::TGLLP},
    {"tgl", SYCLSupportedIntelArchs::TGL},
    {"rkl", SYCLSupportedIntelArchs::RKL},
    {"adl_s", SYCLSupportedIntelArchs::ADL_S},
    {"rpl_s", SYCLSupportedIntelArchs::RPL_S},
    {"adl_p", SYCLSupportedIntelArchs::ADL_P},
    {"adl_n", SYCLSupportedIntelArchs::ADL_N},
    {"dg1", SYCLSupportedIntelArchs::DG1},
    {"acm_g10", SYCLSupportedIntelArchs::ACM_G10},
    {"dg2_g10", SYCLSupportedIntelArchs::DG2_G10},
    {"acm_g11", SYCLSupportedIntelArchs::ACM_G11},
    {"dg2_g10", SYCLSupportedIntelArchs::DG2_G10},
    {"dg2_g11", SYCLSupportedIntelArchs::DG2_G11},
    {"acm_g12", SYCLSupportedIntelArchs::ACM_G12},
    {"dg2_g12", SYCLSupportedIntelArchs::DG2_G12},
    {"pvc", SYCLSupportedIntelArchs::PVC},
    {"pvc_vg", SYCLSupportedIntelArchs::PVC_VG},
    {"mtl_u", SYCLSupportedIntelArchs::MTL_U},
    {"mtl_s", SYCLSupportedIntelArchs::MTL_S},
    {"arl_u", SYCLSupportedIntelArchs::ARL_U},
    {"arl_s", SYCLSupportedIntelArchs::ARL_S},
    {"mtl_h", SYCLSupportedIntelArchs::MTL_H},
    {"arl_h", SYCLSupportedIntelArchs::ARL_H},
    {"bmg_g21", SYCLSupportedIntelArchs::BMG_G21},
    {"lnl_m", SYCLSupportedIntelArchs::LNL_M}};

// Check if the user provided value for --offload-arch is a valid
// SYCL supported Intel AOT target.
SYCLSupportedIntelArchs
clang::driver::StringToOffloadArchSYCL(llvm::StringRef ArchNameAsString) {
  auto result = std::find_if(
      std::begin(StringToArchNamesMap), std::end(StringToArchNamesMap),
      [ArchNameAsString](const StringToOffloadArchSYCLMap &map) {
        return ArchNameAsString == map.ArchName;
      });
  if (result == std::end(StringToArchNamesMap))
    return SYCLSupportedIntelArchs::UNKNOWN;
  return result->IntelArch;
}

// This is a mapping between the user provided --offload-arch value for Intel
// GPU targets and the spir64_gen device name accepted by OCLOC (the Intel GPU
// AOT compiler).
StringRef clang::driver::mapIntelGPUArchName(StringRef ArchName) {
  StringRef Arch;
  Arch = llvm::StringSwitch<StringRef>(ArchName)
             .Case("bdw", "bdw")
             .Case("skl", "skl")
             .Case("kbl", "kbl")
             .Case("cfl", "cfl")
             .Cases("apl", "bxt", "apl")
             .Case("glk", "glk")
             .Case("whl", "whl")
             .Case("aml", "aml")
             .Case("cml", "cml")
             .Cases("icllp", "icl", "icllp")
             .Cases("ehl", "jsl", "ehl")
             .Cases("tgllp", "tgl", "tgllp")
             .Case("rkl", "rkl")
             .Cases("adl_s", "rpl_s", "adl_s")
             .Case("adl_p", "adl_p")
             .Case("adl_n", "adl_n")
             .Case("dg1", "dg1")
             .Cases("acm_g10", "dg2_g10", "acm_g10")
             .Cases("acm_g11", "dg2_g11", "acm_g11")
             .Cases("acm_g12", "dg2_g12", "acm_g12")
             .Case("pvc", "pvc")
             .Case("pvc_vg", "pvc_vg")
             .Cases("mtl_u", "mtl_s", "arl_u", "arl_s", "mtl_u")
             .Case("mtl_h", "mtl_h")
             .Case("arl_h", "arl_h")
             .Case("bmg_g21", "bmg_g21")
             .Case("lnl_m", "lnl_m")
             .Default("");
  return Arch;
}

SYCLInstallationDetector::SYCLInstallationDetector(const Driver &D)
    : D(D), InstallationCandidates() {
  InstallationCandidates.emplace_back(D.Dir + "/..");
}

SYCLInstallationDetector::SYCLInstallationDetector(
    const Driver &D, const llvm::Triple &HostTriple,
    const llvm::opt::ArgList &Args)
    : SYCLInstallationDetector(D) {}

static llvm::SmallString<64>
getLibSpirvBasename(const llvm::Triple &DeviceTriple,
                    const llvm::Triple &HostTriple) {
  // Select remangled libclc variant.
  // Decide long size based on host triple, because offloading targets are going
  // to match that.
  // All known windows environments except Cygwin use 32-bit long.
  llvm::SmallString<64> Result(HostTriple.isOSWindows() &&
                                       !HostTriple.isWindowsCygwinEnvironment()
                                   ? "remangled-l32-signed_char.libspirv-"
                                   : "remangled-l64-signed_char.libspirv-");

  Result.append(DeviceTriple.getTriple());
  Result.append(".bc");

  return Result;
}

const char *SYCLInstallationDetector::findLibspirvPath(
    const llvm::Triple &DeviceTriple, const llvm::opt::ArgList &Args,
    const llvm::Triple &HostTriple) const {

  // If -fsycl-libspirv-path= is specified, try to use that path directly.
  if (Arg *A = Args.getLastArg(options::OPT_fsycl_libspirv_path_EQ)) {
    if (llvm::sys::fs::exists(A->getValue()))
      return A->getValue();

    return nullptr;
  }

  const SmallString<64> Basename =
      getLibSpirvBasename(DeviceTriple, HostTriple);
  auto searchAt = [&](StringRef Path, const Twine &a = "", const Twine &b = "",
                      const Twine &c = "") -> const char * {
    SmallString<128> LibraryPath(Path);
    llvm::sys::path::append(LibraryPath, a, b, c, Basename);

    if (Args.hasArgNoClaim(options::OPT__HASH_HASH_HASH) ||
        llvm::sys::fs::exists(LibraryPath))
      return Args.MakeArgString(LibraryPath);

    return nullptr;
  };

  for (const auto &IC : InstallationCandidates) {
    // Expected path w/out install.
    if (const char *R = searchAt(IC, "lib", "clc"))
      return R;

    // Expected path w/ install.
    if (const char *R = searchAt(IC, "share", "clc"))
      return R;
  }

  return nullptr;
}

void SYCLInstallationDetector::addLibspirvLinkArgs(
    const llvm::Triple &DeviceTriple, const llvm::opt::ArgList &DriverArgs,
    const llvm::Triple &HostTriple, llvm::opt::ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_fno_sycl_libspirv)) {
    // -fno-sycl-libspirv flag is reserved for very unusual cases where the
    // libspirv library is not linked when required by the device: so output
    // appropriate warnings.
    D.Diag(diag::warn_flag_no_sycl_libspirv) << DeviceTriple.str();
    return;
  }

  if (const char *LibSpirvFile =
          findLibspirvPath(DeviceTriple, DriverArgs, HostTriple)) {
    CC1Args.push_back("-mlink-builtin-bitcode");
    CC1Args.push_back(LibSpirvFile);
    return;
  }

  D.Diag(diag::err_drv_no_sycl_libspirv)
      << getLibSpirvBasename(DeviceTriple, HostTriple);
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

void SYCLInstallationDetector::addSYCLIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(clang::driver::options::OPT_nobuiltininc))
    return;
  // Add the SYCL header search locations in the specified order.
  //   ../include/sycl/stl_wrappers
  //   ../include
  SmallString<128> IncludePath(D.Dir);
  llvm::sys::path::append(IncludePath, "..");
  llvm::sys::path::append(IncludePath, "include");
  // This is used to provide our wrappers around STL headers that provide
  // additional functions/template specializations when the user includes those
  // STL headers in their programs (e.g., <complex>).
  SmallString<128> STLWrappersPath(IncludePath);
  llvm::sys::path::append(STLWrappersPath, "sycl");
  llvm::sys::path::append(STLWrappersPath, "stl_wrappers");
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(STLWrappersPath));
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(IncludePath));
}

void SYCLInstallationDetector::print(llvm::raw_ostream &OS) const {
  if (!InstallationCandidates.size())
    return;
  OS << "SYCL Installation Candidates: \n";
  for (const auto &IC : InstallationCandidates) {
    OS << IC << "\n";
  }
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

  // If save-offload-code is passed, put the PTX files
  // into the path provided in save-offload-code.
  if (T->getToolChain().getTriple().isNVPTX() &&
      C.getDriver().isSaveOffloadCodeEnabled() && Ext == "s") {
    SmallString<128> OutputDir;

    Arg *SaveOffloadCodeArg =
        C.getArgs().getLastArg(options::OPT_save_offload_code_EQ);

    OutputDir = (SaveOffloadCodeArg ? SaveOffloadCodeArg->getValue() : "");

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
  C.addCommand(std::move(Cmd));
}

bool SYCL::shouldDoPerObjectFileLinking(const Compilation &C) {
  return !C.getArgs().hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                              /*default=*/true);
}

// Return whether to use native bfloat16 library.
static bool selectBfloatLibs(const llvm::Triple &Triple, const Compilation &C,
                             bool &UseNative) {

  static llvm::SmallSet<StringRef, 8> GPUArchsWithNBF16{
      "intel_gpu_pvc",     "intel_gpu_acm_g10", "intel_gpu_acm_g11",
      "intel_gpu_acm_g12", "intel_gpu_dg2_g10", "intel_gpu_dg2_g11",
      "intel_dg2_g12",     "intel_gpu_bmg_g21", "intel_gpu_lnl_m"};
  const llvm::opt::ArgList &Args = C.getArgs();
  bool NeedLibs = false;

  // spir64 target is actually JIT compilation, so we defer selection of
  // bfloat16 libraries to runtime. For AOT we need libraries, but skip
  // for Nvidia and AMD.
  NeedLibs = Triple.getSubArch() != llvm::Triple::NoSubArch &&
             !Triple.isNVPTX() && !Triple.isAMDGCN();
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

    // We need to select fallback/native bfloat16 devicelib in AOT compilation
    // targetting for Intel GPU devices. Users have 2 ways to apply AOT,
    // 1). clang++ -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc,...,"
    // 2). clang++ -fsycl -fsycl-targets=intel_gpu_pvc,...
    // 3). clang++ -fsycl -fsycl-targets=spir64_gen,intel_gpu_pvc,...
    // -Xsycl-target-backend=spir64_gen "-device dg2"

    std::string Params;
    for (const auto &Arg : TargArgs) {
      Params += " ";
      Params += Arg;
    }

    auto checkBF = [](StringRef Device) {
      return Device.starts_with("pvc") || Device.starts_with("ats") ||
             Device.starts_with("dg2") || Device.starts_with("bmg") ||
             Device.starts_with("lnl");
    };

    auto checkSpirvJIT = [](StringRef Target) {
      return Target.starts_with("spir64-") || Target.starts_with("spirv64-") ||
             (Target == "spir64") || (Target == "spirv64");
    };

    size_t DevicesPos = Params.find("-device ");
    // "-device xxx" is used to specify AOT target device, so user must apply
    // -Xs "-device xxx" or -Xsycl-target-backend=spir64_gen "-device xxx"
    if (DevicesPos != std::string::npos) {
      UseNative = true;
      std::istringstream Devices(Params.substr(DevicesPos + 8));
      for (std::string S; std::getline(Devices, S, ',');)
        UseNative &= checkBF(S);

      // When "-device XXX" is applied to specify GPU type, user can still
      // add -fsycl-targets=intel_gpu_pvc..., native bfloat16 devicelib can
      // only be linked when all GPU types specified support.
      // We need to filter CPU target here and only focus on GPU device.
      if (Arg *SYCLTarget = Args.getLastArg(options::OPT_fsycl_targets_EQ)) {
        for (auto TargetsV : SYCLTarget->getValues()) {
          if (!checkSpirvJIT(StringRef(TargetsV)) &&
              !StringRef(TargetsV).starts_with("spir64_gen") &&
              !StringRef(TargetsV).starts_with("spir64_x86_64") &&
              !GPUArchsWithNBF16.contains(StringRef(TargetsV))) {
            UseNative = false;
            break;
          }
        }
      }

      return NeedLibs;

    } else {
      // -fsycl-targets=intel_gpu_xxx is used to specify AOT target device.
      // Multiple Intel GPU devices can be specified, native bfloat16 devicelib
      // can be involved only when all GPU deivces specified support native
      // bfloat16 native conversion.
      UseNative = true;

      if (Arg *SYCLTarget = Args.getLastArg(options::OPT_fsycl_targets_EQ)) {
        for (auto TargetsV : SYCLTarget->getValues()) {
          if (!checkSpirvJIT(StringRef(TargetsV)) &&
              !GPUArchsWithNBF16.contains(StringRef(TargetsV))) {
            UseNative = false;
            break;
          }
        }
      }
      return NeedLibs;
    }
  }
  return NeedLibs;
}

struct OclocInfo {
  const char *DeviceName;
  const char *PackageName;
  const char *Version;
  SmallVector<int, 8> HexValues;
};

// The PVCDevices data structure is organized by device name, with the
// corresponding ocloc split release, version and possible Hex representations
// of various PVC devices.  This information is gathered from the following:
// https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/devices_base.inl
// https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/devices_additional.inl
static OclocInfo PVCDevices[] = {
    {"pvc-sdv", "gen12+", "12.60.1", {}},
    {"pvc",
     "gen12+",
     "12.60.7",
     {0x0BD0, 0x0BD5, 0x0BD6, 0x0BD7, 0x0BD8, 0x0BD9, 0x0BDA, 0x0BDB}}};

static std::string getDeviceArg(const ArgStringList &CmdArgs) {
  bool DeviceSeen = false;
  std::string DeviceArg;
  for (StringRef Arg : CmdArgs) {
    // -device <arg> comes in as a single arg, split up all potential space
    // separated values.
    SmallVector<StringRef> SplitArgs;
    Arg.split(SplitArgs, ' ');
    for (StringRef SplitArg : SplitArgs) {
      if (DeviceSeen) {
        DeviceArg = SplitArg.str();
        break;
      }
      if (SplitArg == "-device")
        DeviceSeen = true;
    }
    if (DeviceSeen)
      break;
  }

  return DeviceArg;
}

static bool checkPVCDevice(std::string SingleArg, std::string &DevArg) {
  // Handle shortened versions.
  bool CheckShortVersion = true;
  for (auto Char : SingleArg) {
    if (!std::isdigit(Char) && Char != '.') {
      CheckShortVersion = false;
      break;
    }
  }
  // Check for device, version or hex (literal values)
  for (unsigned int I = 0; I < std::size(PVCDevices); I++) {
    if (StringRef(SingleArg).equals_insensitive(PVCDevices[I].DeviceName) ||
        StringRef(SingleArg).equals_insensitive(PVCDevices[I].Version)) {
      DevArg = SingleArg;
      return true;
    }

    for (int HexVal : PVCDevices[I].HexValues) {
      int Value = 0;
      if (!StringRef(SingleArg).getAsInteger(0, Value) && Value == HexVal) {
        // TODO: Pass back the hex string to use for -device_options when
        // IGC is updated to allow.  Currently -device_options only accepts
        // the device ID (i.e. pvc) or the version (12.60.7).
        return true;
      }
    }
    if (CheckShortVersion &&
        StringRef(PVCDevices[I].Version).starts_with(SingleArg)) {
      DevArg = SingleArg;
      return true;
    }
  }

  return false;
}

SmallVector<std::string, 8>
SYCL::getDeviceLibraries(const Compilation &C, const llvm::Triple &TargetTriple,
                         bool IsSpirvAOT) {
  SmallVector<std::string, 8> LibraryList;
  const llvm::opt::ArgList &Args = C.getArgs();

  // For NVPTX and AMDGCN we only use one single bitcode library and ignore
  // manually specified SYCL device libraries.
  bool IgnoreSingleLibs = TargetTriple.isNVPTX() || TargetTriple.isAMDGCN();

  struct DeviceLibOptInfo {
    StringRef DeviceLibName;
    StringRef DeviceLibOption;
  };

  enum { JIT = 0, AOT_CPU, AOT_DG2, AOT_PVC };

  // Currently, all SYCL device libraries will be linked by default.
  llvm::StringMap<bool> DeviceLibLinkInfo = {
      {"libc", true},          {"libm-fp32", true},   {"libm-fp64", true},
      {"libimf-fp32", true},   {"libimf-fp64", true}, {"libimf-bf16", true},
      {"libm-bfloat16", true}, {"internal", true}};

  // If -fno-sycl-device-lib is specified, its values will be used to exclude
  // linkage of libraries specified by DeviceLibLinkInfo. Linkage of "internal"
  // libraries cannot be affected via -fno-sycl-device-lib.
  bool ExcludeDeviceLibs = false;

  if (Arg *A = Args.getLastArg(options::OPT_fsycl_device_lib_EQ,
                               options::OPT_fno_sycl_device_lib_EQ)) {
    if (A->getValues().size() == 0)
      C.getDriver().Diag(diag::warn_drv_empty_joined_argument)
          << A->getAsString(Args);
    else {
      if (A->getOption().matches(options::OPT_fno_sycl_device_lib_EQ))
        ExcludeDeviceLibs = true;

      // When single libraries are ignored and a subset of library names
      // not containing the value "all" is specified by -fno-sycl-device-lib,
      // print an unused argument warning.
      bool PrintUnusedExcludeWarning = false;

      for (StringRef Val : A->getValues()) {
        if (Val == "all") {
          PrintUnusedExcludeWarning = false;

          // Make sure that internal libraries are still linked against
          // when -fno-sycl-device-lib contains "all" and single libraries
          // should be ignored.
          IgnoreSingleLibs = IgnoreSingleLibs && !ExcludeDeviceLibs;

          for (const auto &K : DeviceLibLinkInfo.keys())
            DeviceLibLinkInfo[K] = (K == "internal") || !ExcludeDeviceLibs;
          break;
        }
        auto LinkInfoIter = DeviceLibLinkInfo.find(Val);
        if (LinkInfoIter == DeviceLibLinkInfo.end() || Val == "internal") {
          // TODO: Move the diagnostic to the SYCL section of
          // Driver::CreateOffloadingDeviceToolChains() to minimize code
          // duplication.
          C.getDriver().Diag(diag::err_drv_unsupported_option_argument)
              << A->getSpelling() << Val;
        }
        DeviceLibLinkInfo[Val] = !ExcludeDeviceLibs;
        PrintUnusedExcludeWarning = IgnoreSingleLibs && ExcludeDeviceLibs;
      }
      if (PrintUnusedExcludeWarning)
        C.getDriver().Diag(diag::warn_drv_unused_argument) << A->getSpelling();
    }
  }

  if (TargetTriple.isNVPTX() && IgnoreSingleLibs)
    LibraryList.push_back(
        Args.MakeArgString("devicelib-nvptx64-nvidia-cuda.bc"));

  if (TargetTriple.isAMDGCN() && IgnoreSingleLibs)
    LibraryList.push_back(Args.MakeArgString("devicelib-amdgcn-amd-amdhsa.bc"));

  if (IgnoreSingleLibs)
    return LibraryList;

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
  const SYCLDeviceLibsList SYCLDeviceAsanLibs = {
      {"libsycl-asan", "internal"},
      {"libsycl-asan-cpu", "internal"},
      {"libsycl-asan-dg2", "internal"},
      {"libsycl-asan-pvc", "internal"}};
  const SYCLDeviceLibsList SYCLDeviceMsanLibs = {
      {"libsycl-msan", "internal"},
      {"libsycl-msan-cpu", "internal"},
      // Currently, we only provide aot msan libdevice for PVC and CPU.
      // For DG2, we just use libsycl-msan as placeholder.
      {"libsycl-msan", "internal"},
      {"libsycl-msan-pvc", "internal"}};
  const SYCLDeviceLibsList SYCLDeviceTsanLibs = {
      {"libsycl-tsan", "internal"},
      {"libsycl-tsan-cpu", "internal"},
      // Currently, we only provide aot tsan libdevice for PVC and CPU.
      // For DG2, we just use libsycl-tsan as placeholder.
      // TODO: replace "libsycl-tsan" with "libsycl-tsan-dg2" when DG2
      // AOT support is added.
      {"libsycl-tsan", "internal"},
      {"libsycl-tsan-pvc", "internal"}};
#endif

  const SYCLDeviceLibsList SYCLNativeCpuDeviceLibs = {
      {"libsycl-nativecpu_utils", "internal"}};

  bool IsWindowsMSVCEnv =
      C.getDefaultToolChain().getTriple().isWindowsMSVCEnvironment();
  bool IsNewOffload = C.getDriver().getUseNewOffloadingDriver();
  StringRef LibSuffix = ".bc";
  if (IsNewOffload)
    // For new offload model, we use packaged .bc files.
    LibSuffix = IsWindowsMSVCEnv ? ".new.obj" : ".new.o";
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
  if (IsSpirvAOT)
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

  // Link in ITT annotations library unless fsycl-no-instrument-device-code
  // is specified. This ensures that we are ABI-compatible with the
  // instrumented device code, which was the default not so long ago.
  if (Args.hasFlag(options::OPT_fsycl_instrument_device_code,
                   options::OPT_fno_sycl_instrument_device_code, true))
    addLibraries(SYCLDeviceAnnotationLibs);

#if !defined(_WIN32)

  auto addSingleLibrary = [&](const DeviceLibOptInfo &Lib) {
    if (!DeviceLibLinkInfo[Lib.DeviceLibOption])
      return;
    SmallString<128> LibName(Lib.DeviceLibName);
    llvm::sys::path::replace_extension(LibName, LibSuffix);
    LibraryList.push_back(Args.MakeArgString(LibName));
  };

  // This function is used to check whether there is only one GPU device
  // (PVC or DG2) specified in AOT compilation mode. If yes, we can use
  // corresponding libsycl-asan-* to improve device sanitizer performance,
  // otherwise stick to fallback device sanitizer library used in  JIT mode.
  auto getSpecificGPUTarget = [](const ArgStringList &CmdArgs) -> size_t {
    std::string DeviceArg = getDeviceArg(CmdArgs);
    if ((DeviceArg.empty()) || (DeviceArg.find(",") != std::string::npos))
      return JIT;

    std::string Temp;
    if (checkPVCDevice(DeviceArg, Temp))
      return AOT_PVC;

    if (DeviceArg == "dg2")
      return AOT_DG2;

    return JIT;
  };

  auto getSingleBuildTarget = [&]() -> size_t {
    if (!IsSpirvAOT)
      return JIT;

    llvm::opt::Arg *SYCLTarget = Args.getLastArg(options::OPT_fsycl_targets_EQ);
    if (!SYCLTarget || (SYCLTarget->getValues().size() != 1))
      return JIT;

    StringRef SYCLTargetStr = SYCLTarget->getValue();
    if (SYCLTargetStr.starts_with("spir64_x86_64"))
      return AOT_CPU;

    if (SYCLTargetStr == "intel_gpu_pvc")
      return AOT_PVC;

    if (SYCLTargetStr.starts_with("intel_gpu_dg2"))
      return AOT_DG2;

    if (SYCLTargetStr.starts_with("spir64_gen")) {
      ArgStringList TargArgs;
      Args.AddAllArgValues(TargArgs, options::OPT_Xs, options::OPT_Xs_separate);
      Args.AddAllArgValues(TargArgs, options::OPT_Xsycl_backend);
      llvm::opt::Arg *A = nullptr;
      if ((A = Args.getLastArg(options::OPT_Xsycl_backend_EQ)) &&
          StringRef(A->getValue()).starts_with("spir64_gen"))
        TargArgs.push_back(A->getValue(1));

      return getSpecificGPUTarget(TargArgs);
    }

    return JIT;
  };

  std::string SanitizeVal;
  size_t sanitizer_lib_idx = getSingleBuildTarget();
  if (Arg *A = Args.getLastArg(options::OPT_fsanitize_EQ,
                               options::OPT_fno_sanitize_EQ)) {
    if (A->getOption().matches(options::OPT_fsanitize_EQ) &&
        A->getValues().size() == 1) {
      SanitizeVal = A->getValue();
    }
  } else {
    // User can pass -fsanitize=address to device compiler via
    // -Xsycl-target-frontend, sanitize device library must be
    // linked with user's device image if so.
    std::vector<std::string> EnabledDeviceSanitizers;

    // NOTE: "-fsanitize=" applies to all device targets
    auto SyclFEArgVals = Args.getAllArgValues(options::OPT_Xsycl_frontend);
    auto SyclFEEQArgVals = Args.getAllArgValues(options::OPT_Xsycl_frontend_EQ);
    auto ArchDeviceVals = Args.getAllArgValues(options::OPT_Xarch_device);

    std::vector<std::string> ArgVals(
        SyclFEArgVals.size() + SyclFEEQArgVals.size() + ArchDeviceVals.size());
    ArgVals.insert(ArgVals.end(), SyclFEArgVals.begin(), SyclFEArgVals.end());
    ArgVals.insert(ArgVals.end(), SyclFEEQArgVals.begin(),
                   SyclFEEQArgVals.end());
    ArgVals.insert(ArgVals.end(), ArchDeviceVals.begin(), ArchDeviceVals.end());

    // Driver will report error if more than one of address sanitizer, memory
    // sanitizer or thread sanitizer is enabled, so we only need to check first
    // one here.
    for (const std::string &Arg : ArgVals) {
      if (Arg.find("-fsanitize=address") != std::string::npos) {
        SanitizeVal = "address";
        break;
      }
      if (Arg.find("-fsanitize=memory") != std::string::npos) {
        SanitizeVal = "memory";
        break;
      }
      if (Arg.find("-fsanitize=thread") != std::string::npos) {
        SanitizeVal = "thread";
        break;
      }
    }
  }

  if (SanitizeVal == "address")
    addSingleLibrary(SYCLDeviceAsanLibs[sanitizer_lib_idx]);
  else if (SanitizeVal == "memory")
    addSingleLibrary(SYCLDeviceMsanLibs[sanitizer_lib_idx]);
  else if (SanitizeVal == "thread")
    addSingleLibrary(SYCLDeviceTsanLibs[sanitizer_lib_idx]);
#endif

  if (TargetTriple.isNativeCPU())
    addLibraries(SYCLNativeCpuDeviceLibs);

  return LibraryList;
}

/// Reads device config file to find information about the SYCL targets in
/// `Targets`, and defines device traits macros accordingly.
void SYCL::populateSYCLDeviceTraitsMacrosArgs(
    Compilation &C, const llvm::opt::ArgList &Args,
    const SmallVectorImpl<std::pair<const ToolChain *, StringRef>> &Targets) {
  if (Targets.empty())
    return;

  const auto &TargetTable = DeviceConfigFile::TargetTable;
  std::map<StringRef, unsigned int> AllDevicesHave;
  std::map<StringRef, bool> AnyDeviceHas;
  bool AnyDeviceHasAnyAspect = false;
  unsigned int ValidTargets = 0;
  for (const auto &[TC, BoundArch] : Targets) {
    assert(TC && "Invalid SYCL Offload Toolchain");
    // Try and find the device arch, if it's empty, try to search for either
    // the whole Triple or just the 'ArchName' string.
    auto TargetIt = TargetTable.end();
    const llvm::Triple &TargetTriple = TC->getTriple();
    const StringRef TargetArch{BoundArch};
    if (!TargetArch.empty()) {
      TargetIt = llvm::find_if(TargetTable, [&](const auto &Value) {
        using namespace tools::SYCL;
        StringRef Device{Value.first};
        if (Device.consume_front(gen::AmdGPU))
          return TargetArch == Device && TargetTriple.isAMDGCN();
        if (Device.consume_front(gen::NvidiaGPU))
          return TargetArch == Device && TargetTriple.isNVPTX();
        if (Device.consume_front(gen::IntelGPU))
          return TargetArch == Device && TargetTriple.isSPIRAOT();
        return TargetArch == Device;
      });
    } else {
      TargetIt = TargetTable.find(TargetTriple.str());
      if (TargetIt == TargetTable.end())
        TargetIt = TargetTable.find(TargetTriple.getArchName().str());
    }

    if (TargetIt != TargetTable.end()) {
      const DeviceConfigFile::TargetInfo &Info = (*TargetIt).second;
      ++ValidTargets;
      const auto &AspectList = Info.aspects;
      const auto &MaySupportOtherAspects = Info.maySupportOtherAspects;
      if (!AnyDeviceHasAnyAspect)
        AnyDeviceHasAnyAspect = MaySupportOtherAspects;
      for (const auto &Aspect : AspectList) {
        // If target has an entry in the config file, the set of aspects
        // supported by all devices supporting the target is 'AspectList'.
        // If there's no entry, such set is empty.
        const auto &AspectIt = AllDevicesHave.find(Aspect);
        if (AspectIt != AllDevicesHave.end())
          ++AllDevicesHave[Aspect];
        else
          AllDevicesHave[Aspect] = 1;
        // If target has an entry in the config file AND
        // 'MaySupportOtherAspects' is false, the set of aspects supported
        // by any device supporting the target is 'AspectList'. If there's
        // no entry OR 'MaySupportOtherAspects' is true, such set contains
        // all the aspects.
        AnyDeviceHas[Aspect] = true;
      }
    }
  }
  // If there's no entry for the target in the device config file, the set
  // of aspects supported by any device supporting the target contains all
  // the aspects.
  if (ValidTargets == 0)
    AnyDeviceHasAnyAspect = true;

  const Driver &D = C.getDriver();
  if (AnyDeviceHasAnyAspect) {
    // There exists some target that supports any given aspect.
    constexpr static StringRef MacroAnyDeviceAnyAspect{
        "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"};
    D.addSYCLDeviceTraitsMacroArg(Args, MacroAnyDeviceAnyAspect);
  } else {
    // Some of the aspects are not supported at all by any of the targets.
    // Thus, we need to define individual macros for each supported aspect.
    for (const auto &[TargetKey, SupportedTarget] : AnyDeviceHas) {
      assert(SupportedTarget);
      const SmallString<64> MacroAnyDevice{
          {"-D__SYCL_ANY_DEVICE_HAS_", TargetKey, "__=1"}};
      D.addSYCLDeviceTraitsMacroArg(Args, MacroAnyDevice);
    }
  }
  for (const auto &[TargetKey, SupportedTargets] : AllDevicesHave) {
    if (SupportedTargets != ValidTargets)
      continue;
    const SmallString<64> MacroAllDevices{
        {"-D__SYCL_ALL_DEVICES_HAVE_", TargetKey, "__=1"}};
    D.addSYCLDeviceTraitsMacroArg(Args, MacroAllDevices);
  }
}

// The list should match pre-built SYCL device library files located in
// compiler package. Once we add or remove any SYCL device library files,
// the list should be updated accordingly.
static llvm::SmallVector<StringRef, 16> SYCLDeviceLibList{
    "bfloat16",
    "crt",
    "cmath",
    "cmath-fp64",
    "complex",
    "complex-fp64",
#if defined(_WIN32)
    "msvc-math",
#else
    "asan",
    "asan-pvc",
    "asan-cpu",
    "asan-dg2",
    "msan",
    "msan-pvc",
    "msan-cpu",
    "tsan",
    "tsan-pvc",
    "tsan-cpu",
#endif
    "imf",
    "imf-fp64",
    "imf-bf16",
    "itt-compiler-wrappers",
    "itt-stubs",
    "itt-user-wrappers",
    "fallback-cassert",
    "fallback-cstring",
    "fallback-cmath",
    "fallback-cmath-fp64",
    "fallback-complex",
    "fallback-complex-fp64",
    "fallback-imf",
    "fallback-imf-fp64",
    "fallback-imf-bf16",
    "fallback-bfloat16",
    "native-bfloat16"};

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
      const bool IsNVPTX = this->getToolChain().getTriple().isNVPTX();
      const bool IsAMDGCN = this->getToolChain().getTriple().isAMDGCN();
      const bool IsSYCLNativeCPU =
          this->getToolChain().getTriple().isNativeCPU();
      StringRef LibPostfix = ".bc";
      StringRef NewLibPostfix = ".new.o";
      if (HostTC->getTriple().isWindowsMSVCEnvironment() &&
          C.getDriver().IsCLMode())
        NewLibPostfix = ".new.obj";
      std::string FileName = this->getToolChain().getInputFilename(II);
      StringRef InputFilename = llvm::sys::path::filename(FileName);
      // NativeCPU links against libclc (libspirv)
      if (IsSYCLNativeCPU && InputFilename.contains("libspirv"))
        return true;
      // AMDGCN links against our libdevice (devicelib)
      if (IsAMDGCN && InputFilename.starts_with("devicelib-"))
        return true;
      // NVPTX links against our libclc (libspirv), our libdevice (devicelib),
      // and the CUDA libdevice
      if (IsNVPTX && (InputFilename.starts_with("devicelib-") ||
                      InputFilename.contains("libspirv") ||
                      InputFilename.contains("libdevice")))
        return true;
      StringRef LibSyclPrefix("libsycl-");
      if (!InputFilename.starts_with(LibSyclPrefix) ||
          !InputFilename.ends_with(LibPostfix) ||
          InputFilename.ends_with(NewLibPostfix))
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
        std::string DeviceLibName(L);
        DeviceLibName.append(LibPostfix);
        if (StringRef(PureLibName) == DeviceLibName ||
            (IsNVPTX && StringRef(PureLibName).starts_with(L)))
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
    if (LinkSYCLDeviceLibs) {
      Opts.push_back("-only-needed");
    }
    // Go through the Inputs to the link.  When a listfile is encountered, we
    // know it is an unbundled generated list.
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

// For SYCL the inputs of the linker job are SPIR-V binaries and output is
// a single SPIR-V binary.  Input can also be bitcode when specified by
// the user.
void SYCL::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {

  assert((getToolChain().getTriple().isSPIROrSPIRV() ||
          getToolChain().getTriple().isNVPTX() ||
          getToolChain().getTriple().isAMDGCN() ||
          getToolChain().getTriple().isNativeCPU()) &&
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

// Determine if any of the given arguments contain any PVC based values for
// the -device option.
static bool hasPVCDevice(const ArgStringList &CmdArgs, std::string &DevArg) {
  std::string Res = getDeviceArg(CmdArgs);
  if (Res.empty())
    return false;
  // Go through all of the arguments to '-device' and determine if any of these
  // are pvc based.  We only match literal values and will not find a match
  // when ranges or wildcards are used.
  // Here we parse the targets, tokenizing via ','
  StringRef DeviceArg(Res.c_str());
  SmallVector<StringRef> SplitArgs;
  DeviceArg.split(SplitArgs, ",");
  for (const auto &SingleArg : SplitArgs) {
    bool IsPVC = checkPVCDevice(SingleArg.str(), DevArg);
    if (IsPVC)
      return true;
  }
  return false;
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
  assert(getToolChain().getTriple().isSPIROrSPIRV() && "Unsupported target");
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
                          *HostTC, Device);
  TC.TranslateBackendTargetArgs(getToolChain().getTriple(), Args, CmdArgs,
                                Device);
  TC.TranslateLinkerTargetArgs(getToolChain().getTriple(), Args, CmdArgs,
                               Device);
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
          .Cases("intel_gpu_icllp", "intel_gpu_icl", "intel_gpu_11_0_0",
                 "icllp")
          .Cases("intel_gpu_ehl", "intel_gpu_jsl", "intel_gpu_11_2_0", "ehl")
          .Cases("intel_gpu_tgllp", "intel_gpu_tgl", "intel_gpu_12_0_0",
                 "tgllp")
          .Cases("intel_gpu_rkl", "intel_gpu_12_1_0", "rkl")
          .Cases("intel_gpu_adl_s", "intel_gpu_rpl_s", "intel_gpu_12_2_0",
                 "adl_s")
          .Cases("intel_gpu_adl_p", "intel_gpu_12_3_0", "adl_p")
          .Cases("intel_gpu_adl_n", "intel_gpu_12_4_0", "adl_n")
          .Cases("intel_gpu_dg1", "intel_gpu_12_10_0", "dg1")
          .Cases("intel_gpu_acm_g10", "intel_gpu_dg2_g10", "intel_gpu_12_55_8",
                 "acm_g10")
          .Cases("intel_gpu_acm_g11", "intel_gpu_dg2_g11", "intel_gpu_12_56_5",
                 "acm_g11")
          .Cases("intel_gpu_acm_g12", "intel_gpu_dg2_g12", "intel_gpu_12_57_0",
                 "acm_g12")
          .Cases("intel_gpu_pvc", "intel_gpu_12_60_7", "pvc")
          .Cases("intel_gpu_pvc_vg", "intel_gpu_12_61_7", "pvc_vg")
          .Cases("intel_gpu_mtl_u", "intel_gpu_mtl_s", "intel_gpu_arl_u",
                 "intel_gpu_arl_s", "intel_gpu_12_70_4", "mtl_u")
          .Cases("intel_gpu_mtl_h", "intel_gpu_12_71_4", "mtl_h")
          .Cases("intel_gpu_arl_h", "intel_gpu_12_74_4", "arl_h")
          .Cases("intel_gpu_bmg_g21", "intel_gpu_20_1_4", "bmg_g21")
          .Cases("intel_gpu_bmg_g31", "intel_gpu_20_2_0", "bmg_g31")
          .Cases("intel_gpu_lnl_m", "intel_gpu_20_4_4", "lnl_m")
          .Cases("intel_gpu_ptl_h", "intel_gpu_30_0_4", "ptl_h")
          .Cases("intel_gpu_ptl_u", "intel_gpu_30_1_1", "ptl_u")
          .Cases("intel_gpu_wcl", "intel_gpu_30_3_0", "wcl")
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
          .Case("nvidia_gpu_sm_90a", "sm_90a")
          .Case("amd_gpu_gfx700", "gfx700")
          .Case("amd_gpu_gfx701", "gfx701")
          .Case("amd_gpu_gfx702", "gfx702")
          .Case("amd_gpu_gfx703", "gfx703")
          .Case("amd_gpu_gfx704", "gfx704")
          .Case("amd_gpu_gfx705", "gfx705")
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
                      .Case("mtl_u", "INTEL_GPU_MTL_U")
                      .Case("mtl_h", "INTEL_GPU_MTL_H")
                      .Case("arl_h", "INTEL_GPU_ARL_H")
                      .Case("bmg_g21", "INTEL_GPU_BMG_G21")
                      .Case("bmg_g31", "INTEL_GPU_BMG_G31")
                      .Case("lnl_m", "INTEL_GPU_LNL_M")
                      .Case("ptl_h", "INTEL_GPU_PTL_H")
                      .Case("ptl_u", "INTEL_GPU_PTL_U")
                      .Case("wcl", "INTEL_GPU_WCL")
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
                      .Case("sm_90a", "NVIDIA_GPU_SM_90A")
                      .Case("gfx700", "AMD_GPU_GFX700")
                      .Case("gfx701", "AMD_GPU_GFX701")
                      .Case("gfx702", "AMD_GPU_GFX702")
                      .Case("gfx703", "AMD_GPU_GFX703")
                      .Case("gfx704", "AMD_GPU_GFX704")
                      .Case("gfx705", "AMD_GPU_GFX705")
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

// Unsupported options for SYCL device compilation.
//  -fcf-protection, -fsanitize, -fprofile-generate, -fprofile-instr-generate
//  -ftest-coverage, -fcoverage-mapping, -fcreate-profile, -fprofile-arcs
//  -fcs-profile-generate, --coverage
static ArrayRef<options::ID> getUnsupportedOpts() {
  static constexpr options::ID UnsupportedOpts[] = {
      options::OPT_fsanitize_EQ,      // -fsanitize
      options::OPT_fcf_protection_EQ, // -fcf-protection
      options::OPT_fprofile_generate,
      options::OPT_fprofile_generate_EQ,
      options::OPT_fno_profile_generate, // -f[no-]profile-generate
      options::OPT_ftest_coverage,
      options::OPT_fno_test_coverage, // -f[no-]test-coverage
      options::OPT_fcoverage_mapping,
      options::OPT_coverage,             // --coverage
      options::OPT_fno_coverage_mapping, // -f[no-]coverage-mapping
      options::OPT_fprofile_instr_generate,
      options::OPT_fprofile_instr_generate_EQ,
      options::OPT_fprofile_arcs,
      options::OPT_fno_profile_arcs,           // -f[no-]profile-arcs
      options::OPT_fno_profile_instr_generate, // -f[no-]profile-instr-generate
      options::OPT_fcreate_profile,            // -fcreate-profile
      options::OPT_fprofile_instr_use,
      options::OPT_fprofile_instr_use_EQ, // -fprofile-instr-use
      options::OPT_fcs_profile_generate,  // -fcs-profile-generate
      options::OPT_fcs_profile_generate_EQ,
      options::OPT_gline_tables_only, // -gline-tables-only
  };
  return UnsupportedOpts;
}

// Currently supported options by SYCL NativeCPU device compilation
static inline bool SupportedByNativeCPU(const llvm::Triple &Triple,
                                        const OptSpecifier &Opt) {
  if (!Triple.isNativeCPU())
    return false;

  switch (Opt.getID()) {
  case options::OPT_fcoverage_mapping:
  case options::OPT_fno_coverage_mapping:
  case options::OPT_fprofile_instr_generate:
  case options::OPT_fprofile_instr_generate_EQ:
  case options::OPT_fno_profile_instr_generate:
    return true;
  }
  return false;
}

SYCLToolChain::SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC),
      SYCLInstallation(D, Triple, Args) {
  // Lookup binaries into the driver directory, this is used to discover any
  // dependent SYCL offload compilation tools.
  getProgramPaths().push_back(getDriver().Dir);

  // Diagnose unsupported options only once.
  for (OptSpecifier Opt : getUnsupportedOpts()) {
    if (const Arg *A = Args.getLastArg(Opt)) {
      // Native CPU can support options unsupported by other targets.
      if (SupportedByNativeCPU(getTriple(), Opt))
        continue;
      // All sanitizer options are not currently supported, except
      // AddressSanitizer and MemorySanitizer and ThreadSanitizer
      if (A->getOption().getID() == options::OPT_fsanitize_EQ &&
          A->getValues().size() == 1) {
        std::string SanitizeVal = A->getValue();
        if (SanitizeVal == "address" || SanitizeVal == "memory" ||
            SanitizeVal == "thread")
          continue;
      }
      D.Diag(clang::diag::warn_drv_unsupported_option_for_target)
          << A->getAsString(Args) << getTriple().str() << 1;
    }
  }
}

void SYCLToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  if (DeviceOffloadingKind == Action::OFK_SYCL &&
      !getTriple().isSPIROrSPIRV()) {
    SYCLInstallation.addLibspirvLinkArgs(getEffectiveTriple(), DriverArgs,
                                         HostTC.getTriple(), CC1Args);
  }
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
        // NativeCPU should allow most normal cpu options.
        if (SupportedByNativeCPU(getTriple(), Opt.getID()))
          continue;
        if (Opt.getID() == options::OPT_fsanitize_EQ &&
            A->getValues().size() == 1) {
          std::string SanitizeVal = A->getValue();
          if (SanitizeVal == "address" || SanitizeVal == "memory" ||
              SanitizeVal == "thread") {
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
  if (const Arg *TargetArg = Args.getLastArg(Opt_EQ)) {
    StringRef Val = TargetArg->getValue();
    if (auto GpuDevice =
            tools::SYCL::gen::isGPUTarget<tools::SYCL::gen::AmdGPU>(Val)) {
      SmallString<64> OffloadArch("--offload-arch=");
      OffloadArch += GpuDevice->data();
      parseTargetOpts(OffloadArch, Args, CmdArgs);
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
void SYCLToolChain::TranslateTargetOpt(const llvm::Triple &Triple,
                                       const llvm::opt::ArgList &Args,
                                       llvm::opt::ArgStringList &CmdArgs,
                                       OptSpecifier Opt, OptSpecifier Opt_EQ,
                                       StringRef Device) const {
  for (auto *A : Args) {
    bool OptNoTriple;
    OptNoTriple = A->getOption().matches(Opt);
    if (A->getOption().matches(Opt_EQ)) {
      const llvm::Triple OptTargetTriple =
          getDriver().getSYCLDeviceTriple(A->getValue(), A);
      // Passing device args: -X<Opt>=<triple> -opt=val.
      StringRef GenDevice = SYCL::gen::resolveGenDevice(A->getValue());
      bool IsGenTriple = Triple.isSPIR() &&
                         Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen;
      if (IsGenTriple) {
        if (Device != GenDevice && !Device.empty())
          continue;
        if (OptTargetTriple != Triple && GenDevice.empty())
          // Triples do not match, but only skip when we know we are not
          // comparing against intel_gpu_*
          continue;
        if (OptTargetTriple == Triple && !Device.empty())
          // Triples match, but we are expecting a specific device to be set.
          continue;
      } else if (OptTargetTriple != Triple)
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
    WarnForDeprecatedBackendOpts(getDriver(), Triple, Device, ArgString, A);
    parseTargetOpts(ArgString, Args, CmdArgs);
    A->claim();
  }
}

void SYCLToolChain::AddImpliedTargetArgs(const llvm::Triple &Triple,
                                         const llvm::opt::ArgList &Args,
                                         llvm::opt::ArgStringList &CmdArgs,
                                         const JobAction &JA,
                                         const ToolChain &HostTC,
                                         StringRef Device) const {
  // Current implied args are for debug information and disabling of
  // optimizations.  They are passed along to the respective areas as follows:
  // Default device AOT: -g -cl-opt-disable
  // Default device JIT: -g (-O0 is handled by the runtime)
  // GEN:  -options "-g -O0"
  // CPU:  "--bo=-g -cl-opt-disable"
  llvm::opt::ArgStringList BeArgs;
  // Per-device argument vector storing the device name and the backend argument
  // string
  llvm::SmallVector<std::pair<StringRef, StringRef>, 16> PerDeviceArgs;
  bool IsGen = Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen;
  bool IsJIT =
      Triple.isSPIROrSPIRV() && Triple.getSubArch() == llvm::Triple::NoSubArch;
  if (IsGen && Args.hasArg(options::OPT_fsycl_fp64_conv_emu))
    BeArgs.push_back("-ze-fp64-gen-conv-emu");
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
      bool IsDefault = RegAllocMode == "default";
      if (RegAllocMode.empty() || DeviceName != "pvc" ||
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
      } else if (IsJIT) {
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
    // pvc:default on Windows and and pvc:auto otherwise when -device pvc is
    // provided by the user.
    ArgStringList TargArgs;
    Args.AddAllArgValues(TargArgs, options::OPT_Xs, options::OPT_Xs_separate);
    Args.AddAllArgValues(TargArgs, options::OPT_Xsycl_backend);
    // For -Xsycl-target-backend=<triple> the triple value is used to push
    // specific options to the matching device compilation using that triple.
    // Scrutinize this to make sure we are only checking the values needed
    // for the current device compilation.
    for (auto *A : Args) {
      if (!A->getOption().matches(options::OPT_Xsycl_backend_EQ))
        continue;
      if (getDriver().getSYCLDeviceTriple(A->getValue()) == Triple)
        TargArgs.push_back(A->getValue(1));
    }
    // Check for any -device settings.
    std::string DevArg;
    if (IsJIT || Device == "pvc" || hasPVCDevice(TargArgs, DevArg)) {
      // The -device option passed in by the user may not be 'pvc'. Use the
      // value provided by the user if it was specified.
      StringRef DeviceName = "pvc";
      if (!DevArg.empty())
        DeviceName = DevArg;
      StringRef BackendOptName = SYCL::gen::getGenGRFFlag("auto");
      if (IsGen)
        PerDeviceArgs.push_back({Args.MakeArgString(DeviceName),
                                 Args.MakeArgString(BackendOptName)});
      else if (IsJIT)
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
      // ocloc has different names for some of the newer architectures;
      // translate them to the apropriate value here.
      DepInfo =
          llvm::StringSwitch<StringRef>(DepInfo)
              .Cases("pvc_vg", "12_61_7", "pvc_xt_c0_vg")
              .Cases("mtl_u", "mtl_s", "arl_u", "arl_s", "12_70_4", "mtl_s")
              .Cases("mtl_h", "12_71_4", "mtl_p")
              .Cases("arl_h", "12_74_4", "xe_lpgplus_b0")
              .Default(DepInfo);
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
    // -foffload-fp32-prec-[sqrt/div]
    if (Args.hasArg(options::OPT_foffload_fp32_prec_div) ||
        Args.hasArg(options::OPT_foffload_fp32_prec_sqrt))
      BeArgs.push_back("-ze-fp32-correctly-rounded-divide-sqrt");
  } else if (IsJIT) {
    // -ftarget-compile-fast JIT
    Args.AddLastArg(BeArgs, options::OPT_ftarget_compile_fast);
    // -foffload-fp32-prec-div JIT
    Args.AddLastArg(BeArgs, options::OPT_foffload_fp32_prec_div);
    // -foffload-fp32-prec-sqrt JIT
    Args.AddLastArg(BeArgs, options::OPT_foffload_fp32_prec_sqrt);
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
  if (Triple.getSubArch() == llvm::Triple::NoSubArch) {
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

    // Do not add the -Xs to the default SYCL triple when we know we have
    // implied the setting.
    if ((A->getOption().matches(options::OPT_Xs) ||
         A->getOption().matches(options::OPT_Xs_separate)) &&
        Triple.getSubArch() == llvm::Triple::NoSubArch &&
        Triple.isSPIROrSPIRV() && getDriver().isSYCLDefaultTripleImplied())
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
  // Do not process -Xsycl-target-backend for implied spir64/spirv64
  if (Triple.getSubArch() == llvm::Triple::NoSubArch &&
      Triple.isSPIROrSPIRV() && getDriver().isSYCLDefaultTripleImplied())
    return;
  // Handle -Xsycl-target-backend.
  TranslateTargetOpt(Triple, Args, CmdArgs, options::OPT_Xsycl_backend,
                     options::OPT_Xsycl_backend_EQ, Device);
  TranslateGPUTargetOpt(Args, CmdArgs, options::OPT_fsycl_targets_EQ);
}

void SYCLToolChain::TranslateLinkerTargetArgs(const llvm::Triple &Triple,
                                              const llvm::opt::ArgList &Args,
                                              llvm::opt::ArgStringList &CmdArgs,
                                              StringRef Device) const {
  // Do not process -Xsycl-target-linker for implied spir64/spirv64
  if (Triple.getSubArch() == llvm::Triple::NoSubArch &&
      Triple.isSPIROrSPIRV() && getDriver().isSYCLDefaultTripleImplied())
    return;
  // Handle -Xsycl-target-linker.
  TranslateTargetOpt(Triple, Args, CmdArgs, options::OPT_Xsycl_linker,
                     options::OPT_Xsycl_linker_EQ, Device);
}

Tool *SYCLToolChain::buildBackendCompiler() const {
  if (getTriple().getSubArch() == llvm::Triple::SPIRSubArch_gen)
    return new tools::SYCL::gen::BackendCompiler(*this);
  // fall through is CPU.
  return new tools::SYCL::x86_64::BackendCompiler(*this);
}

Tool *SYCLToolChain::buildLinker() const {
  assert(getTriple().isSPIROrSPIRV() || getTriple().isNativeCPU());
  return new tools::SYCL::Linker(*this);
}

void SYCLToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
SYCLToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void SYCLToolChain::addSYCLIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  SYCLInstallation.addSYCLIncludeArgs(DriverArgs, CC1Args);
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
  return SanitizerKind::Address | SanitizerKind::Memory | SanitizerKind::Thread;
}
