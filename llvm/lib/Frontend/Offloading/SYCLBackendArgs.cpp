//===- SYCLBackendArgs.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Offloading/SYCLBackendArgs.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/Arg.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"

#include <cctype>
#include <cstring>

using namespace llvm;
using namespace llvm::opt;
using namespace llvm::offloading::sycl;

// Mirrors clang/lib/Driver/ToolChains/SYCL.cpp's GRFModeFlagMap.
static const StringMap<StringRef> &getGRFModeFlagMap() {
  static const StringMap<StringRef> Map = {
      {"auto", "-ze-intel-enable-auto-large-GRF-mode"},
      {"small", "-ze-intel-128-GRF-per-thread"},
      {"large", "-ze-opt-large-register-file"}};
  return Map;
}

StringRef llvm::offloading::sycl::getGenGRFFlag(StringRef GRFMode) {
  const auto &Map = getGRFModeFlagMap();
  auto It = Map.find(GRFMode);
  if (It == Map.end())
    return "";
  return It->getValue();
}

StringRef llvm::offloading::sycl::resolveGenDevice(StringRef DeviceName) {
  // Mirror of the StringSwitch table in
  // clang/lib/Driver/ToolChains/SYCL.cpp's SYCL::gen::resolveGenDevice.
  // Keep in sync with that function.
  StringRef Device;
  Device =
      llvm::StringSwitch<StringRef>(DeviceName)
          .Cases({"intel_gpu_bdw", "intel_gpu_8_0_0"}, "bdw")
          .Cases({"intel_gpu_skl", "intel_gpu_9_0_9"}, "skl")
          .Cases({"intel_gpu_kbl", "intel_gpu_9_1_9"}, "kbl")
          .Cases({"intel_gpu_cfl", "intel_gpu_9_2_9"}, "cfl")
          .Cases({"intel_gpu_apl", "intel_gpu_bxt", "intel_gpu_9_3_0"}, "apl")
          .Cases({"intel_gpu_glk", "intel_gpu_9_4_0"}, "glk")
          .Cases({"intel_gpu_whl", "intel_gpu_9_5_0"}, "whl")
          .Cases({"intel_gpu_aml", "intel_gpu_9_6_0"}, "aml")
          .Cases({"intel_gpu_cml", "intel_gpu_9_7_0"}, "cml")
          .Cases({"intel_gpu_icllp", "intel_gpu_icl", "intel_gpu_11_0_0"},
                 "icllp")
          .Cases({"intel_gpu_ehl", "intel_gpu_jsl", "intel_gpu_11_2_0"}, "ehl")
          .Cases({"intel_gpu_tgllp", "intel_gpu_tgl", "intel_gpu_12_0_0"},
                 "tgllp")
          .Cases({"intel_gpu_rkl", "intel_gpu_12_1_0"}, "rkl")
          .Cases({"intel_gpu_adl_s", "intel_gpu_rpl_s", "intel_gpu_12_2_0"},
                 "adl_s")
          .Cases({"intel_gpu_adl_p", "intel_gpu_12_3_0"}, "adl_p")
          .Cases({"intel_gpu_adl_n", "intel_gpu_12_4_0"}, "adl_n")
          .Cases({"intel_gpu_dg1", "intel_gpu_12_10_0"}, "dg1")
          .Cases(
              {"intel_gpu_acm_g10", "intel_gpu_dg2_g10", "intel_gpu_12_55_8"},
              "acm_g10")
          .Cases(
              {"intel_gpu_acm_g11", "intel_gpu_dg2_g11", "intel_gpu_12_56_5"},
              "acm_g11")
          .Cases(
              {"intel_gpu_acm_g12", "intel_gpu_dg2_g12", "intel_gpu_12_57_0"},
              "acm_g12")
          .Cases({"intel_gpu_pvc", "intel_gpu_12_60_7"}, "pvc")
          .Cases({"intel_gpu_pvc_vg", "intel_gpu_12_61_7"}, "pvc_vg")
          .Cases({"intel_gpu_mtl_u", "intel_gpu_mtl_s", "intel_gpu_arl_u",
                  "intel_gpu_arl_s", "intel_gpu_12_70_4"},
                 "mtl_u")
          .Cases({"intel_gpu_mtl_h", "intel_gpu_12_71_4"}, "mtl_h")
          .Cases({"intel_gpu_arl_h", "intel_gpu_12_74_4"}, "arl_h")
          .Cases({"intel_gpu_bmg_g21", "intel_gpu_20_1_4"}, "bmg_g21")
          .Cases({"intel_gpu_bmg_g31", "intel_gpu_20_2_0"}, "bmg_g31")
          .Cases({"intel_gpu_lnl_m", "intel_gpu_20_4_4"}, "lnl_m")
          .Cases({"intel_gpu_ptl_h", "intel_gpu_30_0_4"}, "ptl_h")
          .Cases({"intel_gpu_ptl_u", "intel_gpu_30_1_1"}, "ptl_u")
          .Cases({"intel_gpu_wcl", "intel_gpu_30_3_0"}, "wcl")
          .Cases({"intel_gpu_nvl_s", "intel_gpu_nvl_hx", "intel_gpu_nvl_ul",
                  "intel_gpu_30_4_0"},
                 "nvl_s")
          .Cases({"intel_gpu_nvl_u", "intel_gpu_nvl_h", "intel_gpu_30_5_0"},
                 "nvl_u")
          .Cases({"intel_gpu_nvl_p", "intel_gpu_35_10_0"}, "nvl_p")
          .Cases({"intel_gpu_cri", "intel_gpu_35_11_0"}, "cri")
          .Case("intel_gpu_dg2", "dg2")
          .Case("intel_gpu_mtl", "mtl")
          .Case("intel_gpu_bmg", "bmg")
          .Case("intel_gpu_ptl", "ptl")
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

void llvm::offloading::sycl::parseTargetOpts(StringRef ArgString,
                                             const ArgList &Args,
                                             ArgStringList &CmdArgs) {
  SmallVector<const char *, 8> TargetArgs;
  BumpPtrAllocator A;
  StringSaver S(A);
  cl::TokenizeGNUCommandLine(ArgString, S, TargetArgs);
  for (StringRef TA : TargetArgs)
    CmdArgs.push_back(Args.MakeArgString(TA));
}

namespace {
// Matches the legacy in-Driver `OclocInfo` table. Kept private to this file.
struct OclocInfo {
  const char *DeviceName;
  const char *PackageName;
  const char *Version;
  ArrayRef<int> HexValues;
};

constexpr int PVCHex[] = {0x0BD0, 0x0BD5, 0x0BD6, 0x0BD7,
                          0x0BD8, 0x0BD9, 0x0BDA, 0x0BDB};
const OclocInfo PVCDevices[] = {
    {"pvc-sdv", "gen12+", "12.60.1", {}},
    {"pvc", "gen12+", "12.60.7", PVCHex},
};

bool checkPVCDevice(StringRef SingleArg, std::string &DevArg) {
  bool CheckShortVersion = true;
  for (char C : SingleArg) {
    if (!std::isdigit(static_cast<unsigned char>(C)) && C != '.') {
      CheckShortVersion = false;
      break;
    }
  }
  for (const OclocInfo &Info : PVCDevices) {
    if (SingleArg.equals_insensitive(Info.DeviceName) ||
        SingleArg.equals_insensitive(Info.Version)) {
      DevArg = SingleArg.str();
      return true;
    }
    for (int HexVal : Info.HexValues) {
      int Value = 0;
      if (!SingleArg.getAsInteger(0, Value) && Value == HexVal)
        return true;
    }
    if (CheckShortVersion && StringRef(Info.Version).starts_with(SingleArg)) {
      DevArg = SingleArg.str();
      return true;
    }
  }
  return false;
}

std::string getDeviceArg(ArrayRef<const char *> CmdArgs) {
  bool DeviceSeen = false;
  std::string DeviceArg;
  for (StringRef Arg : CmdArgs) {
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
} // namespace

bool llvm::offloading::sycl::hasPVCDevice(ArrayRef<const char *> CmdArgs,
                                          std::string &DevArg) {
  std::string Res = getDeviceArg(CmdArgs);
  if (Res.empty())
    return false;
  StringRef DeviceArg(Res);
  SmallVector<StringRef> SplitArgs;
  DeviceArg.split(SplitArgs, ",");
  for (StringRef SingleArg : SplitArgs) {
    if (checkPVCDevice(SingleArg, DevArg))
      return true;
  }
  return false;
}

namespace {
// Mirrors `WarnForDeprecatedBackendOpts` from SYCL.cpp: warn when a user is
// using a deprecated ocloc GRF backend opt against PVC.
void warnForDeprecatedBackendOpts(const BackendArgsInput &In, StringRef Device,
                                  StringRef ArgString, const Arg *A) {
  if (!ArgString.contains("-device pvc") && !Device.contains("pvc"))
    return;
  // Only warn the second time around for gen targets; the translators run
  // twice and the device is set on the second pass.
  if (In.Triple.isSPIR() &&
      In.Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen &&
      !A->isClaimed())
    return;
  for (const auto &KV : getGRFModeFlagMap()) {
    StringRef Mode = KV.getKey();
    StringRef Flag = KV.getValue();
    if (ArgString.contains(Flag) && In.WarnPVCDeprecatedGRFFn)
      In.WarnPVCDeprecatedGRFFn(Flag, Mode);
  }
}
} // namespace

void llvm::offloading::sycl::translateTargetOpt(const BackendArgsInput &In,
                                                OptSpecifier Opt,
                                                OptSpecifier Opt_EQ,
                                                ArgStringList &CmdArgs) {
  for (Arg *A : In.Args) {
    bool OptNoTriple = A->getOption().matches(Opt);
    if (A->getOption().matches(Opt_EQ)) {
      const llvm::Triple OptTargetTriple =
          In.ResolveDeviceTriple ? In.ResolveDeviceTriple(A->getValue(), A)
                                 : llvm::Triple(A->getValue());
      StringRef GenDevice = resolveGenDevice(A->getValue());
      bool IsGenTriple =
          In.Triple.isSPIR() &&
          In.Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen;
      if (IsGenTriple) {
        if (In.Device != GenDevice && !In.Device.empty())
          continue;
        if (OptTargetTriple != In.Triple && GenDevice.empty())
          continue;
        if (OptTargetTriple == In.Triple && !In.Device.empty())
          continue;
      } else if (OptTargetTriple != In.Triple)
        continue;
    } else if (!OptNoTriple)
      continue;

    StringRef ArgString;
    if (OptNoTriple) {
      // With multiple -fsycl-targets a triple is required so we know where
      // the option should go.
      if (const Arg *TargetArg = In.Args.getLastArg(In.Ids.offload_targets_EQ))
        if (TargetArg->getValues().size() != 1) {
          if (In.ErrXsyclTargetMissingTripleFn)
            In.ErrXsyclTargetMissingTripleFn(A->getSpelling());
          continue;
        }
      ArgString = A->getValue();
    } else
      ArgString = A->getValue(1);
    warnForDeprecatedBackendOpts(In, In.Device, ArgString, A);
    parseTargetOpts(ArgString, In.Args, CmdArgs);
    A->claim();
  }
}

void llvm::offloading::sycl::translateGPUTargetOpt(const BackendArgsInput &In,
                                                   OptSpecifier Opt_EQ,
                                                   ArgStringList &CmdArgs) {
  if (const Arg *TargetArg = In.Args.getLastArg(Opt_EQ)) {
    StringRef Val = TargetArg->getValue();
    // The legacy code calls `tools::SYCL::gen::isGPUTarget<AmdGPU>(Val)`,
    // which is a one-line wrapper around starts_with("amd_gpu_") plus
    // resolveGenDevice. Inline it here to avoid pulling in the Driver
    // helper.
    constexpr StringRef AmdGPUPrefix = "amd_gpu_";
    if (Val.starts_with(AmdGPUPrefix)) {
      StringRef GpuDevice = resolveGenDevice(Val);
      if (!GpuDevice.empty()) {
        SmallString<64> OffloadArch("--offload-arch=");
        OffloadArch += GpuDevice;
        parseTargetOpts(OffloadArch, In.Args, CmdArgs);
      }
    }
  }
}

void llvm::offloading::sycl::translateBackendTargetArgs(
    const BackendArgsInput &In, ArgStringList &CmdArgs) {
  // Handle -Xs flags.
  for (Arg *A : In.Args) {
    bool IsXs = A->getOption().matches(In.Ids.Xs);
    bool IsXsSep = A->getOption().matches(In.Ids.Xs_separate);
    if ((IsXs || IsXsSep) &&
        In.Triple.getSubArch() == llvm::Triple::NoSubArch &&
        In.Triple.isSPIROrSPIRV() && In.IsSYCLDefaultTripleImplied)
      continue;

    if (IsXs) {
      CmdArgs.push_back(In.Args.MakeArgString(Twine("-") + A->getValue()));
      warnForDeprecatedBackendOpts(In, In.Device, A->getValue(), A);
      A->claim();
      continue;
    }
    if (IsXsSep) {
      StringRef ArgString(A->getValue());
      parseTargetOpts(ArgString, In.Args, CmdArgs);
      warnForDeprecatedBackendOpts(In, In.Device, ArgString, A);
      A->claim();
      continue;
    }
  }
  // Do not process -Xsycl-target-backend for implied spir64/spirv64.
  if (In.Triple.getSubArch() == llvm::Triple::NoSubArch &&
      In.Triple.isSPIROrSPIRV() && In.IsSYCLDefaultTripleImplied)
    return;
  translateTargetOpt(In, In.Ids.Xsycl_backend, In.Ids.Xsycl_backend_EQ,
                     CmdArgs);
  translateGPUTargetOpt(In, In.Ids.offload_targets_EQ, CmdArgs);
}

void llvm::offloading::sycl::translateLinkerTargetArgs(
    const BackendArgsInput &In, ArgStringList &CmdArgs) {
  if (In.Triple.getSubArch() == llvm::Triple::NoSubArch &&
      In.Triple.isSPIROrSPIRV() && In.IsSYCLDefaultTripleImplied)
    return;
  translateTargetOpt(In, In.Ids.Xsycl_linker, In.Ids.Xsycl_linker_EQ, CmdArgs);
}

void llvm::offloading::sycl::addSPIRVImpliedTargetArgs(
    const BackendArgsInput &In, ArgStringList &CmdArgs) {
  // Implied args for debug info and -O0:
  //   Default device AOT: -g -cl-opt-disable
  //   Default device JIT: -g  (-O0 is handled by the runtime)
  //   GEN:                -options "-g -O0"
  //   CPU:                "--bo=-g" "-bo=-cl-opt-disable"
  ArgStringList BeArgs;
  SmallVector<std::pair<StringRef, StringRef>, 16> PerDeviceArgs;
  bool IsGen = In.Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen;
  bool IsJIT = In.Triple.isSPIROrSPIRV() &&
               In.Triple.getSubArch() == llvm::Triple::NoSubArch;

  if (IsGen && In.Args.hasArg(In.Ids.fsycl_fp64_conv_emu))
    BeArgs.push_back("-ze-fp64-gen-conv-emu");

  if (Arg *A = In.Args.getLastArg(In.Ids.g_Group, In.Ids._SLASH_Z7))
    if (!A->getOption().matches(In.Ids.g0))
      BeArgs.push_back("-g");

  if (In.Triple.getSubArch() != llvm::Triple::NoSubArch)
    if (Arg *A = In.Args.getLastArg(In.Ids.O_Group))
      if (A->getOption().matches(In.Ids.O0))
        BeArgs.push_back("-cl-opt-disable");

  StringRef RegAllocModeOptName = "-ftarget-register-alloc-mode=";
  if (Arg *A = In.Args.getLastArg(In.Ids.ftarget_register_alloc_mode_EQ)) {
    StringRef RegAllocModeVal = A->getValue(0);
    auto ProcessElement = [&](StringRef Ele) {
      auto [DeviceName, RegAllocMode] = Ele.split(':');
      StringRef BackendOptName = getGenGRFFlag(RegAllocMode);
      bool IsDefault = RegAllocMode == "default";
      if (RegAllocMode.empty() || DeviceName != "pvc" ||
          (BackendOptName.empty() && !IsDefault)) {
        if (In.ErrUnsupportedOptionArgumentFn)
          In.ErrUnsupportedOptionArgumentFn(A->getSpelling(), Ele);
      }
      if (IsDefault)
        return;
      if (IsGen) {
        PerDeviceArgs.push_back(
            {DeviceName, In.Args.MakeArgString(BackendOptName)});
      } else if (IsJIT) {
        BeArgs.push_back(In.Args.MakeArgString(
            RegAllocModeOptName + DeviceName + ":" + BackendOptName));
      }
    };
    SmallVector<StringRef, 16> RegAllocModeArgs;
    RegAllocModeVal.split(RegAllocModeArgs, ',');
    for (StringRef Elem : RegAllocModeArgs)
      ProcessElement(Elem);
  } else if (!In.HostIsWindowsMSVCEnv) {
    // Default register-alloc-mode = pvc:auto on non-Windows when -device pvc
    // is in play.
    ArgStringList TargArgs;
    In.Args.AddAllArgValues(TargArgs, In.Ids.Xs, In.Ids.Xs_separate);
    In.Args.AddAllArgValues(TargArgs, In.Ids.Xsycl_backend);
    for (Arg *A : In.Args) {
      if (!A->getOption().matches(In.Ids.Xsycl_backend_EQ))
        continue;
      llvm::Triple ResolvedTriple =
          In.ResolveDeviceTriple ? In.ResolveDeviceTriple(A->getValue(), A)
                                 : llvm::Triple(A->getValue());
      if (ResolvedTriple == In.Triple)
        TargArgs.push_back(A->getValue(1));
    }
    std::string DevArg;
    if (IsJIT || In.Device == "pvc" || hasPVCDevice(TargArgs, DevArg)) {
      StringRef DeviceName = "pvc";
      if (!DevArg.empty())
        DeviceName = DevArg;
      StringRef BackendOptName = getGenGRFFlag("auto");
      if (IsGen)
        PerDeviceArgs.push_back({In.Args.MakeArgString(DeviceName),
                                 In.Args.MakeArgString(BackendOptName)});
      else if (IsJIT)
        BeArgs.push_back(In.Args.MakeArgString(
            RegAllocModeOptName + DeviceName + ":" + BackendOptName));
    }
  }

  if (IsGen) {
    // intel_gpu_<arch> implied -device for AOT GEN.
    StringRef DepInfo = In.JobOffloadingArch;
    if (!DepInfo.empty()) {
      ArgStringList TargArgs;
      In.Args.AddAllArgValues(TargArgs, In.Ids.Xs, In.Ids.Xs_separate);
      In.Args.AddAllArgValues(TargArgs, In.Ids.Xsycl_backend);
      for (Arg *A : In.Args) {
        if (!A->getOption().matches(In.Ids.Xsycl_backend_EQ))
          continue;
        if (StringRef(A->getValue()).starts_with("intel_gpu"))
          TargArgs.push_back(A->getValue(1));
      }
      if (llvm::find_if(TargArgs, [&](const char *Cur) {
            return !std::strncmp(Cur, "-device", sizeof("-device") - 1);
          }) != TargArgs.end()) {
        if (In.ErrUnsupportedOptForTargetFn) {
          SmallString<64> Target("intel_gpu_");
          Target += DepInfo;
          In.ErrUnsupportedOptForTargetFn("-device", Target);
        }
      }
      // ocloc renames for newer architectures.
      DepInfo =
          StringSwitch<StringRef>(DepInfo)
              .Cases({"pvc_vg", "12_61_7"}, "pvc_xt_c0_vg")
              .Cases({"mtl_u", "mtl_s", "arl_u", "arl_s", "12_70_4"}, "mtl_s")
              .Cases({"mtl_h", "12_71_4"}, "mtl_p")
              .Cases({"arl_h", "12_74_4"}, "xe_lpgplus_b0")
              .Default(DepInfo);
      CmdArgs.push_back("-device");
      CmdArgs.push_back(In.Args.MakeArgString(DepInfo));
    }
    // -ftarget-compile-fast (AOT)
    if (In.Args.hasArg(In.Ids.ftarget_compile_fast))
      BeArgs.push_back("-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'");
    // -ftarget-export-symbols
    if (In.Args.hasFlag(In.Ids.ftarget_export_symbols,
                        In.Ids.fno_target_export_symbols, false))
      BeArgs.push_back("-library-compilation");
    // -foffload-fp32-prec-{div,sqrt}
    if (In.Args.hasArg(In.Ids.foffload_fp32_prec_div) ||
        In.Args.hasArg(In.Ids.foffload_fp32_prec_sqrt))
      BeArgs.push_back("-ze-fp32-correctly-rounded-divide-sqrt");
  } else if (IsJIT) {
    // -ftarget-compile-fast (JIT)
    In.Args.AddLastArg(BeArgs, In.Ids.ftarget_compile_fast);
    // -foffload-fp32-prec-{div,sqrt} (JIT)
    In.Args.AddLastArg(BeArgs, In.Ids.foffload_fp32_prec_div);
    In.Args.AddLastArg(BeArgs, In.Ids.foffload_fp32_prec_sqrt);
  }

  if (IsGen) {
    for (auto [DeviceName, BackendArgStr] : PerDeviceArgs) {
      CmdArgs.push_back("-device_options");
      CmdArgs.push_back(In.Args.MakeArgString(DeviceName));
      CmdArgs.push_back(In.Args.MakeArgString(BackendArgStr));
    }
  }
  if (BeArgs.empty())
    return;
  if (In.Triple.getSubArch() == llvm::Triple::NoSubArch) {
    for (StringRef A : BeArgs)
      CmdArgs.push_back(In.Args.MakeArgString(A));
    return;
  }
  if (IsGen) {
    SmallString<128> BeOpt;
    CmdArgs.push_back("-options");
    for (unsigned I = 0; I < BeArgs.size(); ++I) {
      if (I)
        BeOpt += ' ';
      BeOpt += BeArgs[I];
    }
    CmdArgs.push_back(In.Args.MakeArgString(BeOpt));
  } else {
    for (StringRef A : BeArgs) {
      SmallString<128> BeOpt;
      BeOpt += "--bo=";
      BeOpt += A;
      CmdArgs.push_back(In.Args.MakeArgString(BeOpt));
    }
  }
}
