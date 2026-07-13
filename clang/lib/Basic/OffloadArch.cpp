//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/OffloadArch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {

namespace {
struct OffloadArchToStringMap {
  OffloadArch Arch;
  const char *ArchName;
  const char *VirtualArchName;
};
} // namespace

#define SM(sm) {OffloadArch::SM_##sm, "sm_" #sm, "compute_" #sm}
#define GFX(gpu) {OffloadArch::GFX##gpu, "gfx" #gpu, "compute_amdgcn"}
static const OffloadArchToStringMap ArchNames[] = {
    // clang-format off
    {OffloadArch::Unused, "", ""},
    SM(20), {OffloadArch::SM_21, "sm_21", "compute_20"}, // Fermi
    SM(30), {OffloadArch::SM_32_, "sm_32", "compute_32"}, SM(35), SM(37),  // Kepler
    SM(50), SM(52), SM(53),          // Maxwell
    SM(60), SM(61), SM(62),          // Pascal
    SM(70), SM(72),                  // Volta
    SM(75),                          // Turing
    SM(80), SM(86),                  // Ampere
    SM(87),                          // Jetson/Drive AGX Orin
    SM(88),                          // Ampere
    SM(89),                          // Ada Lovelace
    SM(90),                          // Hopper
    SM(90a),                         // Hopper
    SM(100),                         // Blackwell
    SM(100a),                        // Blackwell
    SM(100f),                        // Blackwell
    SM(101),                         // Blackwell
    SM(101a),                        // Blackwell
    SM(101f),                        // Blackwell
    SM(103),                         // Blackwell
    SM(103a),                        // Blackwell
    SM(103f),                        // Blackwell
    SM(110),                         // Blackwell
    SM(110a),                        // Blackwell
    SM(110f),                        // Blackwell
    SM(120),                         // Blackwell
    SM(120a),                        // Blackwell
    SM(120f),                        // Blackwell
    SM(121),                         // Blackwell
    SM(121a),                        // Blackwell
    SM(121f),                        // Blackwell
    GFX(600),  // gfx600
    GFX(601),  // gfx601
    GFX(602),  // gfx602
    GFX(700),  // gfx700
    GFX(701),  // gfx701
    GFX(702),  // gfx702
    GFX(703),  // gfx703
    GFX(704),  // gfx704
    GFX(705),  // gfx705
    GFX(801),  // gfx801
    GFX(802),  // gfx802
    GFX(803),  // gfx803
    GFX(805),  // gfx805
    GFX(810),  // gfx810
    {OffloadArch::GFX9_GENERIC, "gfx9-generic", "compute_amdgcn"},
    GFX(900),  // gfx900
    GFX(902),  // gfx902
    GFX(904),  // gfx903
    GFX(906),  // gfx906
    GFX(908),  // gfx908
    GFX(909),  // gfx909
    GFX(90a),  // gfx90a
    GFX(90c),  // gfx90c
    {OffloadArch::GFX9_4_GENERIC, "gfx9-4-generic", "compute_amdgcn"},
    GFX(942),  // gfx942
    GFX(950),  // gfx950
    {OffloadArch::GFX10_1_GENERIC, "gfx10-1-generic", "compute_amdgcn"},
    GFX(1010), // gfx1010
    GFX(1011), // gfx1011
    GFX(1012), // gfx1012
    GFX(1013), // gfx1013
    {OffloadArch::GFX10_3_GENERIC, "gfx10-3-generic", "compute_amdgcn"},
    GFX(1030), // gfx1030
    GFX(1031), // gfx1031
    GFX(1032), // gfx1032
    GFX(1033), // gfx1033
    GFX(1034), // gfx1034
    GFX(1035), // gfx1035
    GFX(1036), // gfx1036
    {OffloadArch::GFX11_GENERIC, "gfx11-generic", "compute_amdgcn"},
    GFX(1100), // gfx1100
    GFX(1101), // gfx1101
    GFX(1102), // gfx1102
    GFX(1103), // gfx1103
    GFX(1150), // gfx1150
    GFX(1151), // gfx1151
    GFX(1152), // gfx1152
    GFX(1153), // gfx1153
    GFX(1154), // gfx1154
    GFX(1170), // gfx1170
    GFX(1171), // gfx1171
    GFX(1172), // gfx1172
    {OffloadArch::GFX12_GENERIC, "gfx12-generic", "compute_amdgcn"},
    GFX(1200), // gfx1200
    GFX(1201), // gfx1201
    GFX(1250), // gfx1250
    GFX(1251), // gfx1251
    {OffloadArch::GFX12_5_GENERIC, "gfx12-5-generic", "compute_amdgcn"},
    GFX(1310), // gfx1310
    {OffloadArch::AMDGCNSPIRV, "amdgcnspirv", "compute_amdgcn"},
    // Intel CPUs
    {OffloadArch::SKYLAKEAVX512, "skylake-avx512", ""},
    {OffloadArch::COREAVX2, "core-avx2", ""},
    {OffloadArch::COREI7AVX, "corei7-avx", ""},
    {OffloadArch::COREI7, "corei7", ""},
    {OffloadArch::WESTMERE, "westmere", ""},
    {OffloadArch::SANDYBRIDGE, "sandybridge", ""},
    {OffloadArch::IVYBRIDGE, "ivybridge", ""},
    {OffloadArch::BROADWELL, "broadwell", ""},
    {OffloadArch::COFFEELAKE, "coffeelake", ""},
    {OffloadArch::ALDERLAKE, "alderlake", ""},
    {OffloadArch::SKYLAKE, "skylake", ""},
    {OffloadArch::SKX, "skx", ""},
    {OffloadArch::CASCADELAKE, "cascadelake", ""},
    {OffloadArch::ICELAKECLIENT, "icelake-client", ""},
    {OffloadArch::ICELAKESERVER, "icelake-server", ""},
    {OffloadArch::SAPPHIRERAPIDS, "sapphirerapids", ""},
    {OffloadArch::GRANITERAPIDS, "graniterapids", ""},
    // Intel GPUS
    {OffloadArch::BDW, "bdw", ""},
    {OffloadArch::SKL, "skl", ""},
    {OffloadArch::KBL, "kbl", ""},
    {OffloadArch::CFL, "cfl", ""},
    {OffloadArch::APL, "apl", ""},
    {OffloadArch::BXT, "bxt", ""},
    {OffloadArch::GLK, "glk", ""},
    {OffloadArch::WHL, "whl", ""},
    {OffloadArch::AML, "aml", ""},
    {OffloadArch::CML, "cml", ""},
    {OffloadArch::ICLLP, "icllp", ""},
    {OffloadArch::ICL, "icl", ""},
    {OffloadArch::EHL, "ehl", ""},
    {OffloadArch::JSL, "jsl", ""},
    {OffloadArch::TGLLP, "tgllp", ""},
    {OffloadArch::TGL, "tgl", ""},
    {OffloadArch::RKL, "rkl", ""},
    {OffloadArch::ADL_S, "adl_s", ""},
    {OffloadArch::RPL_S, "rpl_s", ""},
    {OffloadArch::ADL_P, "adl_p", ""},
    {OffloadArch::ADL_N, "adl_n", ""},
    {OffloadArch::DG1, "dg1", ""},
    {OffloadArch::DG2, "dg2", ""},
    {OffloadArch::ACM_G10, "acm_g10", ""},
    {OffloadArch::DG2_G10, "dg2_g10", ""},
    {OffloadArch::ACM_G11, "acm_g11", ""},
    {OffloadArch::DG2_G11, "dg2_g11", ""},
    {OffloadArch::ACM_G12, "acm_g12", ""},
    {OffloadArch::DG2_G12, "dg2_g12", ""},
    {OffloadArch::PVC, "pvc", ""},
    {OffloadArch::PVC_VG, "pvc_vg", ""},
    {OffloadArch::MTL, "mtl", ""},
    {OffloadArch::MTL_U, "mtl_u", ""},
    {OffloadArch::MTL_S, "mtl_s", ""},
    {OffloadArch::ARL_U, "arl_u", ""},
    {OffloadArch::ARL_S, "arl_s", ""},
    {OffloadArch::MTL_H, "mtl_h", ""},
    {OffloadArch::ARL_H, "arl_h", ""},
    {OffloadArch::BMG, "bmg", ""},
    {OffloadArch::BMG_G21, "bmg_g21", ""},
    {OffloadArch::PTL, "ptl", ""},
    {OffloadArch::LNL_M, "lnl_m", ""},
    {OffloadArch::Generic, "generic", ""},
    // clang-format on
};
#undef SM
#undef GFX

const char *OffloadArchToString(OffloadArch A) {
  auto Result =
      llvm::find_if(ArchNames, [A](const OffloadArchToStringMap &Map) {
        return A == Map.Arch;
      });
  if (Result == std::end(ArchNames))
    return "unknown";
  return Result->ArchName;
}

const char *OffloadArchToVirtualArchString(OffloadArch A) {
  auto Result =
      llvm::find_if(ArchNames, [A](const OffloadArchToStringMap &Map) {
        return A == Map.Arch;
      });
  if (Result == std::end(ArchNames))
    return "unknown";
  return Result->VirtualArchName;
}

OffloadArch StringToOffloadArch(llvm::StringRef S) {
  auto Result =
      llvm::find_if(ArchNames, [S](const OffloadArchToStringMap &Map) {
        return S == Map.ArchName;
      });
  if (Result == std::end(ArchNames))
    return OffloadArch::Unknown;
  return Result->Arch;
}

llvm::Triple OffloadArchToTriple(const llvm::Triple &DefaultToolchainTriple,
                                 OffloadArch ID) {
  if (ID == OffloadArch::AMDGCNSPIRV)
    return llvm::Triple(llvm::Triple::spirv64, llvm::Triple::NoSubArch,
                        llvm::Triple::AMD, llvm::Triple::AMDHSA);

  if (IsNVIDIAOffloadArch(ID)) {
    llvm::Triple::ArchType Arch = DefaultToolchainTriple.isArch64Bit()
                                      ? llvm::Triple::nvptx64
                                      : llvm::Triple::nvptx;
    return llvm::Triple(Arch, llvm::Triple::NoSubArch, llvm::Triple::NVIDIA,
                        llvm::Triple::CUDA);
  }

  if (IsAMDOffloadArch(ID))
    return llvm::Triple(llvm::Triple::amdgcn, llvm::Triple::NoSubArch,
                        llvm::Triple::AMD, llvm::Triple::AMDHSA);

  if (IsIntelCPUOffloadArch(ID))
    return llvm::Triple("spir64_x86_64-unknown-unknown");

  if (IsIntelGPUOffloadArch(ID))
    return llvm::Triple("spir64_gen-unknown-unknown");

  return {};
}

llvm::StringRef SYCLTargetToOffloadArchName(llvm::StringRef SYCLTarget) {
  return llvm::StringSwitch<llvm::StringRef>(SYCLTarget)
      // Intel GPU – symbolic names
      .Cases({"intel_gpu_bdw", "intel_gpu_8_0_0"}, "bdw")
      .Cases({"intel_gpu_skl", "intel_gpu_9_0_9"}, "skl")
      .Cases({"intel_gpu_kbl", "intel_gpu_9_1_9"}, "kbl")
      .Cases({"intel_gpu_cfl", "intel_gpu_9_2_9"}, "cfl")
      .Cases({"intel_gpu_apl", "intel_gpu_bxt", "intel_gpu_9_3_0"}, "apl")
      .Cases({"intel_gpu_glk", "intel_gpu_9_4_0"}, "glk")
      .Cases({"intel_gpu_whl", "intel_gpu_9_5_0"}, "whl")
      .Cases({"intel_gpu_aml", "intel_gpu_9_6_0"}, "aml")
      .Cases({"intel_gpu_cml", "intel_gpu_9_7_0"}, "cml")
      .Cases({"intel_gpu_icllp", "intel_gpu_icl", "intel_gpu_11_0_0"}, "icllp")
      .Cases({"intel_gpu_ehl", "intel_gpu_jsl", "intel_gpu_11_2_0"}, "ehl")
      .Cases({"intel_gpu_tgllp", "intel_gpu_tgl", "intel_gpu_12_0_0"}, "tgllp")
      .Cases({"intel_gpu_rkl", "intel_gpu_12_1_0"}, "rkl")
      .Cases({"intel_gpu_adl_s", "intel_gpu_rpl_s", "intel_gpu_12_2_0"},
             "adl_s")
      .Cases({"intel_gpu_adl_p", "intel_gpu_12_3_0"}, "adl_p")
      .Cases({"intel_gpu_adl_n", "intel_gpu_12_4_0"}, "adl_n")
      .Cases({"intel_gpu_dg1", "intel_gpu_12_10_0"}, "dg1")
      .Cases({"intel_gpu_acm_g10", "intel_gpu_dg2_g10", "intel_gpu_12_55_8"},
             "acm_g10")
      .Cases({"intel_gpu_acm_g11", "intel_gpu_dg2_g11", "intel_gpu_12_56_5"},
             "acm_g11")
      .Cases({"intel_gpu_acm_g12", "intel_gpu_dg2_g12", "intel_gpu_12_57_0"},
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
      // NVIDIA GPU
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
      // AMD GPU
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
}

llvm::StringRef NormalizeIntelGPUOffloadArch(llvm::StringRef ArchName) {
  return llvm::StringSwitch<llvm::StringRef>(ArchName)
      .Case("bdw", "bdw")
      .Case("skl", "skl")
      .Case("kbl", "kbl")
      .Case("cfl", "cfl")
      .Cases({"apl", "bxt"}, "apl")
      .Case("glk", "glk")
      .Case("whl", "whl")
      .Case("aml", "aml")
      .Case("cml", "cml")
      .Cases({"icllp", "icl"}, "icllp")
      .Cases({"ehl", "jsl"}, "ehl")
      .Cases({"tgllp", "tgl"}, "tgllp")
      .Case("rkl", "rkl")
      .Cases({"adl_s", "rpl_s"}, "adl_s")
      .Case("adl_p", "adl_p")
      .Case("adl_n", "adl_n")
      .Case("dg1", "dg1")
      .Cases({"acm_g10", "dg2_g10"}, "acm_g10")
      .Cases({"acm_g11", "dg2_g11"}, "acm_g11")
      .Cases({"acm_g12", "dg2_g12"}, "acm_g12")
      .Case("pvc", "pvc")
      .Case("pvc_vg", "pvc_vg")
      .Cases({"mtl_u", "mtl_s", "arl_u", "arl_s"}, "mtl_u")
      .Case("mtl_h", "mtl_h")
      .Case("arl_h", "arl_h")
      .Case("bmg_g21", "bmg_g21")
      .Case("lnl_m", "lnl_m")
      .Default(ArchName);
}

llvm::StringRef GetOffloadArchMacroSuffix(llvm::StringRef DeviceName) {
  llvm::StringRef Ext =
      llvm::StringSwitch<llvm::StringRef>(DeviceName)
          // Intel GPU
          .Case("bdw", "INTEL_GPU_BDW")
          .Case("skl", "INTEL_GPU_SKL")
          .Case("kbl", "INTEL_GPU_KBL")
          .Case("cfl", "INTEL_GPU_CFL")
          .Cases({"apl", "bxt"}, "INTEL_GPU_APL")
          .Case("glk", "INTEL_GPU_GLK")
          .Case("whl", "INTEL_GPU_WHL")
          .Case("aml", "INTEL_GPU_AML")
          .Case("cml", "INTEL_GPU_CML")
          .Cases({"icllp", "icl"}, "INTEL_GPU_ICLLP")
          .Cases({"ehl", "jsl"}, "INTEL_GPU_EHL")
          .Cases({"tgllp", "tgl"}, "INTEL_GPU_TGLLP")
          .Case("rkl", "INTEL_GPU_RKL")
          .Cases({"adl_s", "rpl_s"}, "INTEL_GPU_ADL_S")
          .Case("adl_p", "INTEL_GPU_ADL_P")
          .Case("adl_n", "INTEL_GPU_ADL_N")
          .Case("dg1", "INTEL_GPU_DG1")
          .Case("dg2", "INTEL_GPU_DG2")
          .Cases({"acm_g10", "dg2_g10"}, "INTEL_GPU_ACM_G10")
          .Cases({"acm_g11", "dg2_g11"}, "INTEL_GPU_ACM_G11")
          .Cases({"acm_g12", "dg2_g12"}, "INTEL_GPU_ACM_G12")
          .Case("pvc", "INTEL_GPU_PVC")
          .Case("pvc_vg", "INTEL_GPU_PVC_VG")
          .Case("mtl", "INTEL_GPU_MTL")
          .Cases({"mtl_u", "mtl_s", "arl_u", "arl_s"}, "INTEL_GPU_MTL_U")
          .Case("mtl_h", "INTEL_GPU_MTL_H")
          .Case("arl_h", "INTEL_GPU_ARL_H")
          .Case("bmg", "INTEL_GPU_BMG")
          .Case("bmg_g21", "INTEL_GPU_BMG_G21")
          .Case("bmg_g31", "INTEL_GPU_BMG_G31")
          .Case("lnl_m", "INTEL_GPU_LNL_M")
          .Case("ptl", "INTEL_GPU_PTL")
          .Case("ptl_h", "INTEL_GPU_PTL_H")
          .Case("ptl_u", "INTEL_GPU_PTL_U")
          .Case("wcl", "INTEL_GPU_WCL")
          .Case("nvl_s", "INTEL_GPU_NVL_S")
          .Case("nvl_u", "INTEL_GPU_NVL_U")
          .Case("nvl_p", "INTEL_GPU_NVL_P")
          .Case("cri", "INTEL_GPU_CRI")
          // NVIDIA GPU
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
          // AMD GPU
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
  return Ext;
}

} // namespace clang
