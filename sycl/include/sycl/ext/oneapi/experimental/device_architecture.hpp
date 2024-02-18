//===- device_architecture.hpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <optional>
#include <unordered_map>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class architecture {
  // If new element is added to this enum:
  //
  // Update
  //   - "detail::min_<category>_architecture" below if needed
  //   - "detail::max_<category>_architecture" below if needed
  //   - sycl_ext_oneapi_device_architecture specification doc
  //   - "-fsycl-targets" description in sycl/doc/UsersManual.md
  //
  // Add
  //   - __SYCL_TARGET_<ARCH>__ to the compiler driver and to all places below
  //   - the unique ID of the new architecture in SYCL RT source code to support
  //     querying the device architecture
  //
  x86_64 = 0x00000000,
  //
  // Intel CPU architectures
  //
  // The requirement for the unique ID for intel_cpu_* architectures below is:
  // - the ID must start with 0x0 (to avoid the integer overflow)
  // - then goes Intel's vendor ID from underlied backend (which is 8086)
  // - the ID ends with the architecture ID from the DEVICE_IP_VERSION extension
  //   of underlied backend
  intel_cpu_spr = 0x08086008,
  intel_cpu_gnr = 0x08086009,
  //
  // Intel GPU architectures
  //
  // The requirement for the unique ID for intel_gpu_* architectures below is:
  // - the ID is GMDID of that architecture
  intel_gpu_bdw = 0x02000000,
  intel_gpu_skl = 0x02400009,
  intel_gpu_kbl = 0x02404009,
  intel_gpu_cfl = 0x02408009,
  intel_gpu_apl = 0x0240c000,
  intel_gpu_bxt = intel_gpu_apl,
  intel_gpu_glk = 0x02410000,
  intel_gpu_whl = 0x02414000,
  intel_gpu_aml = 0x02418000,
  intel_gpu_cml = 0x0241c000,
  intel_gpu_icllp = 0x02c00000,
  intel_gpu_ehl = 0x02c08000,
  intel_gpu_jsl = intel_gpu_ehl,
  intel_gpu_tgllp = 0x03000000,
  intel_gpu_rkl = 0x03004000,
  intel_gpu_adl_s = 0x03008000,
  intel_gpu_rpl_s = intel_gpu_adl_s,
  intel_gpu_adl_p = 0x0300c000,
  intel_gpu_adl_n = 0x03010000,
  intel_gpu_dg1 = 0x03028000,
  intel_gpu_acm_g10 = 0x030dc008,
  intel_gpu_dg2_g10 = intel_gpu_acm_g10,
  intel_gpu_acm_g11 = 0x030e0005,
  intel_gpu_dg2_g11 = intel_gpu_acm_g11,
  intel_gpu_acm_g12 = 0x030e4000,
  intel_gpu_dg2_g12 = intel_gpu_acm_g12,
  intel_gpu_pvc = 0x030f0007,
  intel_gpu_pvc_vg = 0x030f4007,
  //
  // NVIDIA architectures
  //
  // The requirement for the unique ID for nvidia_gpu_* architectures below is:
  // - the ID must start with NVIDIA's vendor ID from underlied backend (which
  //   is 0x10de)
  // - the ID must end with SM version ID of that architecture
  nvidia_gpu_sm_50 = 0x10de0050,
  nvidia_gpu_sm_52 = 0x10de0052,
  nvidia_gpu_sm_53 = 0x10de0053,
  nvidia_gpu_sm_60 = 0x10de0060,
  nvidia_gpu_sm_61 = 0x10de0061,
  nvidia_gpu_sm_62 = 0x10de0062,
  nvidia_gpu_sm_70 = 0x10de0070,
  nvidia_gpu_sm_72 = 0x10de0072,
  nvidia_gpu_sm_75 = 0x10de0075,
  nvidia_gpu_sm_80 = 0x10de0080,
  nvidia_gpu_sm_86 = 0x10de0086,
  nvidia_gpu_sm_87 = 0x10de0087,
  nvidia_gpu_sm_89 = 0x10de0089,
  nvidia_gpu_sm_90 = 0x10de0090,
  //
  // AMD architectures
  //
  // The requirement for the unique ID for amd_gpu_* architectures below is:
  // - the ID must start with AMD's vendor ID from underlied backend (which is
  //   0x1002)
  // - the ID must end with GFX version ID of that architecture
  amd_gpu_gfx700 = 0x10020700,
  amd_gpu_gfx701 = 0x10020701,
  amd_gpu_gfx702 = 0x10020702,
  amd_gpu_gfx801 = 0x10020801,
  amd_gpu_gfx802 = 0x10020802,
  amd_gpu_gfx803 = 0x10020803,
  amd_gpu_gfx805 = 0x10020805,
  amd_gpu_gfx810 = 0x10020810,
  amd_gpu_gfx900 = 0x10020900,
  amd_gpu_gfx902 = 0x10020902,
  amd_gpu_gfx904 = 0x10020904,
  amd_gpu_gfx906 = 0x10020906,
  amd_gpu_gfx908 = 0x10020908,
  amd_gpu_gfx909 = 0x10020909,
  amd_gpu_gfx90a = 0x1002090a,
  amd_gpu_gfx90c = 0x1002090c,
  amd_gpu_gfx940 = 0x10020940,
  amd_gpu_gfx941 = 0x10020941,
  amd_gpu_gfx942 = 0x10020942,
  amd_gpu_gfx1010 = 0x10021010,
  amd_gpu_gfx1011 = 0x10021011,
  amd_gpu_gfx1012 = 0x10021012,
  amd_gpu_gfx1013 = 0x10021013,
  amd_gpu_gfx1030 = 0x10021030,
  amd_gpu_gfx1031 = 0x10021031,
  amd_gpu_gfx1032 = 0x10021032,
  amd_gpu_gfx1033 = 0x10021033,
  amd_gpu_gfx1034 = 0x10021034,
  amd_gpu_gfx1035 = 0x10021035,
  amd_gpu_gfx1036 = 0x10021036,
  amd_gpu_gfx1100 = 0x10021100,
  amd_gpu_gfx1101 = 0x10021101,
  amd_gpu_gfx1102 = 0x10021102,
  amd_gpu_gfx1103 = 0x10021103,
  amd_gpu_gfx1150 = 0x10021150,
  amd_gpu_gfx1151 = 0x10021151,
  amd_gpu_gfx1200 = 0x10021200,
  amd_gpu_gfx1201 = 0x10021201,
  intel_gpu_8_0_0 = intel_gpu_bdw,
  intel_gpu_9_0_9 = intel_gpu_skl,
  intel_gpu_9_1_9 = intel_gpu_kbl,
  intel_gpu_9_2_9 = intel_gpu_cfl,
  intel_gpu_9_3_0 = intel_gpu_apl,
  intel_gpu_9_4_0 = intel_gpu_glk,
  intel_gpu_9_5_0 = intel_gpu_whl,
  intel_gpu_9_6_0 = intel_gpu_aml,
  intel_gpu_9_7_0 = intel_gpu_cml,
  intel_gpu_11_0_0 = intel_gpu_icllp,
  intel_gpu_12_0_0 = intel_gpu_tgllp,
  intel_gpu_12_10_0 = intel_gpu_dg1,
};

enum class arch_category {
  intel_gpu = 0,
  nvidia_gpu = 1,
  amd_gpu = 2,
  // TODO: add intel_cpu
};

} // namespace ext::oneapi::experimental

namespace detail {

static constexpr ext::oneapi::experimental::architecture
    min_intel_gpu_architecture =
        ext::oneapi::experimental::architecture::intel_gpu_bdw;
static constexpr ext::oneapi::experimental::architecture
    max_intel_gpu_architecture =
        ext::oneapi::experimental::architecture::intel_gpu_pvc_vg;

static constexpr ext::oneapi::experimental::architecture
    min_nvidia_gpu_architecture =
        ext::oneapi::experimental::architecture::nvidia_gpu_sm_50;
static constexpr ext::oneapi::experimental::architecture
    max_nvidia_gpu_architecture =
        ext::oneapi::experimental::architecture::nvidia_gpu_sm_90;

static constexpr ext::oneapi::experimental::architecture
    min_amd_gpu_architecture =
        ext::oneapi::experimental::architecture::amd_gpu_gfx700;
static constexpr ext::oneapi::experimental::architecture
    max_amd_gpu_architecture =
        ext::oneapi::experimental::architecture::amd_gpu_gfx1201;

#ifndef __SYCL_TARGET_INTEL_X86_64__
#define __SYCL_TARGET_INTEL_X86_64__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_BDW__
#define __SYCL_TARGET_INTEL_GPU_BDW__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_SKL__
#define __SYCL_TARGET_INTEL_GPU_SKL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_KBL__
#define __SYCL_TARGET_INTEL_GPU_KBL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_CFL__
#define __SYCL_TARGET_INTEL_GPU_CFL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_APL__
#define __SYCL_TARGET_INTEL_GPU_APL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_GLK__
#define __SYCL_TARGET_INTEL_GPU_GLK__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_WHL__
#define __SYCL_TARGET_INTEL_GPU_WHL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_AML__
#define __SYCL_TARGET_INTEL_GPU_AML__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_CML__
#define __SYCL_TARGET_INTEL_GPU_CML__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ICLLP__
#define __SYCL_TARGET_INTEL_GPU_ICLLP__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_EHL__
#define __SYCL_TARGET_INTEL_GPU_EHL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_TGLLP__
#define __SYCL_TARGET_INTEL_GPU_TGLLP__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_RKL__
#define __SYCL_TARGET_INTEL_GPU_RKL__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ADL_S__
#define __SYCL_TARGET_INTEL_GPU_ADL_S__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ADL_P__
#define __SYCL_TARGET_INTEL_GPU_ADL_P__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ADL_N__
#define __SYCL_TARGET_INTEL_GPU_ADL_N__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_DG1__
#define __SYCL_TARGET_INTEL_GPU_DG1__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ACM_G10__
#define __SYCL_TARGET_INTEL_GPU_ACM_G10__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ACM_G11__
#define __SYCL_TARGET_INTEL_GPU_ACM_G11__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ACM_G12__
#define __SYCL_TARGET_INTEL_GPU_ACM_G12__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_PVC__
#define __SYCL_TARGET_INTEL_GPU_PVC__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_PVC_VG__
#define __SYCL_TARGET_INTEL_GPU_PVC_VG__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM50__
#define __SYCL_TARGET_NVIDIA_GPU_SM50__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM52__
#define __SYCL_TARGET_NVIDIA_GPU_SM52__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM53__
#define __SYCL_TARGET_NVIDIA_GPU_SM53__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM60__
#define __SYCL_TARGET_NVIDIA_GPU_SM60__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM61__
#define __SYCL_TARGET_NVIDIA_GPU_SM61__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM62__
#define __SYCL_TARGET_NVIDIA_GPU_SM62__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM70__
#define __SYCL_TARGET_NVIDIA_GPU_SM70__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM72__
#define __SYCL_TARGET_NVIDIA_GPU_SM72__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM75__
#define __SYCL_TARGET_NVIDIA_GPU_SM75__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM80__
#define __SYCL_TARGET_NVIDIA_GPU_SM80__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM86__
#define __SYCL_TARGET_NVIDIA_GPU_SM86__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM87__
#define __SYCL_TARGET_NVIDIA_GPU_SM87__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM89__
#define __SYCL_TARGET_NVIDIA_GPU_SM89__ 0
#endif
#ifndef __SYCL_TARGET_NVIDIA_GPU_SM90__
#define __SYCL_TARGET_NVIDIA_GPU_SM90__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX700__
#define __SYCL_TARGET_AMD_GPU_GFX700__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX701__
#define __SYCL_TARGET_AMD_GPU_GFX701__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX702__
#define __SYCL_TARGET_AMD_GPU_GFX702__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX801__
#define __SYCL_TARGET_AMD_GPU_GFX801__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX802__
#define __SYCL_TARGET_AMD_GPU_GFX802__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX803__
#define __SYCL_TARGET_AMD_GPU_GFX803__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX805__
#define __SYCL_TARGET_AMD_GPU_GFX805__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX810__
#define __SYCL_TARGET_AMD_GPU_GFX810__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX900__
#define __SYCL_TARGET_AMD_GPU_GFX900__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX902__
#define __SYCL_TARGET_AMD_GPU_GFX902__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX904__
#define __SYCL_TARGET_AMD_GPU_GFX904__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX906__
#define __SYCL_TARGET_AMD_GPU_GFX906__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX908__
#define __SYCL_TARGET_AMD_GPU_GFX908__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX909__
#define __SYCL_TARGET_AMD_GPU_GFX909__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX90A__
#define __SYCL_TARGET_AMD_GPU_GFX90A__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX90C__
#define __SYCL_TARGET_AMD_GPU_GFX90C__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX940__
#define __SYCL_TARGET_AMD_GPU_GFX940__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX941__
#define __SYCL_TARGET_AMD_GPU_GFX941__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX942__
#define __SYCL_TARGET_AMD_GPU_GFX942__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1010__
#define __SYCL_TARGET_AMD_GPU_GFX1010__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1011__
#define __SYCL_TARGET_AMD_GPU_GFX1011__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1012__
#define __SYCL_TARGET_AMD_GPU_GFX1012__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1013__
#define __SYCL_TARGET_AMD_GPU_GFX1013__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1030__
#define __SYCL_TARGET_AMD_GPU_GFX1030__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1031__
#define __SYCL_TARGET_AMD_GPU_GFX1031__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1032__
#define __SYCL_TARGET_AMD_GPU_GFX1032__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1033__
#define __SYCL_TARGET_AMD_GPU_GFX1033__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1034__
#define __SYCL_TARGET_AMD_GPU_GFX1034__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1035__
#define __SYCL_TARGET_AMD_GPU_GFX1035__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1036__
#define __SYCL_TARGET_AMD_GPU_GFX1036__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1100__
#define __SYCL_TARGET_AMD_GPU_GFX1100__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1101__
#define __SYCL_TARGET_AMD_GPU_GFX1101__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1102__
#define __SYCL_TARGET_AMD_GPU_GFX1102__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1103__
#define __SYCL_TARGET_AMD_GPU_GFX1103__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1150__
#define __SYCL_TARGET_AMD_GPU_GFX1150__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1151__
#define __SYCL_TARGET_AMD_GPU_GFX1151__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1200__
#define __SYCL_TARGET_AMD_GPU_GFX1200__ 0
#endif
#ifndef __SYCL_TARGET_AMD_GPU_GFX1201__
#define __SYCL_TARGET_AMD_GPU_GFX1201__ 0
#endif

// This is true when the translation unit is compiled in AOT mode with target
// names that supports the "if_architecture_is" features.  If an unsupported
// target name is specified via "-fsycl-targets", the associated invocation of
// the device compiler will set this variable to false, and that will trigger
// an error for code that uses "if_architecture_is".
static constexpr bool is_allowable_aot_mode =
    (__SYCL_TARGET_INTEL_X86_64__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_BDW__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_SKL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_KBL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_CFL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_APL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_GLK__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_WHL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_AML__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_CML__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ICLLP__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_EHL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_TGLLP__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_RKL__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ADL_S__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ADL_P__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ADL_N__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_DG1__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ACM_G10__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ACM_G11__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ACM_G12__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_PVC__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_PVC_VG__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM50__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM52__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM53__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM60__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM61__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM62__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM70__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM72__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM75__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM80__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM86__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM87__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM89__ == 1) ||
    (__SYCL_TARGET_NVIDIA_GPU_SM90__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX700__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX701__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX702__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX801__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX802__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX803__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX805__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX810__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX900__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX902__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX904__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX906__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX908__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX909__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX90A__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX90C__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX940__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX941__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX942__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1010__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1011__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1012__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1013__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1030__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1031__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1032__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1033__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1034__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1035__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1036__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1100__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1101__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1102__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1103__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1150__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1151__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1200__ == 1) ||
    (__SYCL_TARGET_AMD_GPU_GFX1201__ == 1);

constexpr static std::optional<ext::oneapi::experimental::architecture>
get_current_architecture_aot() {
  // TODO: re-write the logic below when sycl_ext_oneapi_device_architecture
  // will support targets more than one in -fsycl-targets
#if __SYCL_TARGET_INTEL_X86_64__
  return ext::oneapi::experimental::architecture::x86_64;
#endif
#if __SYCL_TARGET_INTEL_GPU_BDW__
  return ext::oneapi::experimental::architecture::intel_gpu_bdw;
#endif
#if __SYCL_TARGET_INTEL_GPU_SKL__
  return ext::oneapi::experimental::architecture::intel_gpu_skl;
#endif
#if __SYCL_TARGET_INTEL_GPU_KBL__
  return ext::oneapi::experimental::architecture::intel_gpu_kbl;
#endif
#if __SYCL_TARGET_INTEL_GPU_CFL__
  return ext::oneapi::experimental::architecture::intel_gpu_cfl;
#endif
#if __SYCL_TARGET_INTEL_GPU_APL__
  return ext::oneapi::experimental::architecture::intel_gpu_apl;
#endif
#if __SYCL_TARGET_INTEL_GPU_GLK__
  return ext::oneapi::experimental::architecture::intel_gpu_glk;
#endif
#if __SYCL_TARGET_INTEL_GPU_WHL__
  return ext::oneapi::experimental::architecture::intel_gpu_whl;
#endif
#if __SYCL_TARGET_INTEL_GPU_AML__
  return ext::oneapi::experimental::architecture::intel_gpu_aml;
#endif
#if __SYCL_TARGET_INTEL_GPU_CML__
  return ext::oneapi::experimental::architecture::intel_gpu_cml;
#endif
#if __SYCL_TARGET_INTEL_GPU_ICLLP__
  return ext::oneapi::experimental::architecture::intel_gpu_icllp;
#endif
#if __SYCL_TARGET_INTEL_GPU_EHL__
  return ext::oneapi::experimental::architecture::intel_gpu_ehl;
#endif
#if __SYCL_TARGET_INTEL_GPU_TGLLP__
  return ext::oneapi::experimental::architecture::intel_gpu_tgllp;
#endif
#if __SYCL_TARGET_INTEL_GPU_RKL__
  return ext::oneapi::experimental::architecture::intel_gpu_rkl;
#endif
#if __SYCL_TARGET_INTEL_GPU_ADL_S__
  return ext::oneapi::experimental::architecture::intel_gpu_adl_s;
#endif
#if __SYCL_TARGET_INTEL_GPU_ADL_P__
  return ext::oneapi::experimental::architecture::intel_gpu_adl_p;
#endif
#if __SYCL_TARGET_INTEL_GPU_ADL_P__
  return ext::oneapi::experimental::architecture::intel_gpu_adl_p;
#endif
#if __SYCL_TARGET_INTEL_GPU_ADL_N__
  return ext::oneapi::experimental::architecture::intel_gpu_adl_n;
#endif
#if __SYCL_TARGET_INTEL_GPU_DG1__
  return ext::oneapi::experimental::architecture::intel_gpu_dg1;
#endif
#if __SYCL_TARGET_INTEL_GPU_ACM_G10__
  return ext::oneapi::experimental::architecture::intel_gpu_acm_g10;
#endif
#if __SYCL_TARGET_INTEL_GPU_ACM_G11__
  return ext::oneapi::experimental::architecture::intel_gpu_acm_g11;
#endif
#if __SYCL_TARGET_INTEL_GPU_ACM_G12__
  return ext::oneapi::experimental::architecture::intel_gpu_acm_g12;
#endif
#if __SYCL_TARGET_INTEL_GPU_PVC__
  return ext::oneapi::experimental::architecture::intel_gpu_pvc;
#endif
#if __SYCL_TARGET_INTEL_GPU_PVC_VG__
  return ext::oneapi::experimental::architecture::intel_gpu_pvc_vg;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM50__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_50;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM52__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_52;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM53__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_53;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM60__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_60;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM61__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_61;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM62__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_62;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM70__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_70;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM72__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_72;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM75__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_75;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM80__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_80;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM86__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_86;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM87__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_87;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM89__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_89;
#endif
#if __SYCL_TARGET_NVIDIA_GPU_SM90__
  return ext::oneapi::experimental::architecture::nvidia_gpu_sm_90;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX700__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx700;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX701__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx701;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX702__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx702;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX801__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx801;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX802__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx802;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX803__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx803;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX805__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx805;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX810__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx810;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX900__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx900;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX902__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx902;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX904__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx904;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX906__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx906;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX908__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx908;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX909__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx909;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX90a__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx90a;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX90c__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx90c;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX940__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx940;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX941__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx941;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX942__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx942;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1010__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1010;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1011__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1011;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1012__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1012;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1030__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1030;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1031__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1031;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1032__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1032;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1033__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1033;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1034__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1034;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1035__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1035;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1036__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1036;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1100__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1100;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1101__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1101;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1102__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1102;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1103__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1103;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1150__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1150;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1151__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1151;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1200__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1200;
#endif
#if __SYCL_TARGET_AMD_GPU_GFX1201__
  return ext::oneapi::experimental::architecture::amd_gpu_gfx1201;
#endif
  return std::nullopt;
}

// Tells if the AOT target matches that architecture.
constexpr static bool
is_aot_for_architecture(ext::oneapi::experimental::architecture arch) {
  constexpr std::optional<ext::oneapi::experimental::architecture>
      current_arch = get_current_architecture_aot();
  if (current_arch.has_value())
    return arch == *current_arch;
  return false;
}

// Reads the value of "is_allowable_aot_mode" via a template to defer triggering
// static_assert() until template instantiation time.
template <ext::oneapi::experimental::architecture... Archs>
constexpr static bool allowable_aot_mode() {
  return is_allowable_aot_mode;
}

// Tells if the current device has one of the architectures in the parameter
// pack.
template <ext::oneapi::experimental::architecture... Archs>
constexpr static bool device_architecture_is() {
  return (is_aot_for_architecture(Archs) || ...);
}

static constexpr std::optional<ext::oneapi::experimental::architecture>
get_category_min_architecture(
    ext::oneapi::experimental::arch_category Category) {
  if (Category == ext::oneapi::experimental::arch_category::intel_gpu) {
    return min_intel_gpu_architecture;
  } else if (Category == ext::oneapi::experimental::arch_category::nvidia_gpu) {
    return min_nvidia_gpu_architecture;
  } else if (Category == ext::oneapi::experimental::arch_category::amd_gpu) {
    return min_amd_gpu_architecture;
  } // add "else if " when adding new category, "else" not needed
  return std::nullopt;
}

static constexpr std::optional<ext::oneapi::experimental::architecture>
get_category_max_architecture(
    ext::oneapi::experimental::arch_category Category) {
  if (Category == ext::oneapi::experimental::arch_category::intel_gpu) {
    return max_intel_gpu_architecture;
  } else if (Category == ext::oneapi::experimental::arch_category::nvidia_gpu) {
    return max_nvidia_gpu_architecture;
  } else if (Category == ext::oneapi::experimental::arch_category::amd_gpu) {
    return max_amd_gpu_architecture;
  } // add "else if " when adding new category, "else" not needed
  return std::nullopt;
}

template <ext::oneapi::experimental::arch_category Category>
constexpr static bool device_architecture_is_in_category_aot() {
  constexpr std::optional<ext::oneapi::experimental::architecture>
      category_min_arch = get_category_min_architecture(Category);
  constexpr std::optional<ext::oneapi::experimental::architecture>
      category_max_arch = get_category_max_architecture(Category);
  constexpr std::optional<ext::oneapi::experimental::architecture>
      current_arch = get_current_architecture_aot();

  if (category_min_arch.has_value() && category_max_arch.has_value() &&
      current_arch.has_value())
    if ((*category_min_arch <= *current_arch) &&
        (*current_arch <= *category_max_arch))
      return true;

  return false;
}

template <ext::oneapi::experimental::arch_category... Categories>
constexpr static bool device_architecture_is_in_categories() {
  return (device_architecture_is_in_category_aot<Categories>() || ...);
}

constexpr static std::optional<ext::oneapi::experimental::arch_category>
get_device_architecture_category(ext::oneapi::experimental::architecture arch) {
  auto arch_is_in_segment =
      [&arch](ext::oneapi::experimental::architecture min,
              ext::oneapi::experimental::architecture max) {
        if ((min <= arch) && (arch <= max))
          return true;
        return false;
      };

  if (arch_is_in_segment(min_intel_gpu_architecture,
                         max_intel_gpu_architecture))
    return ext::oneapi::experimental::arch_category::intel_gpu;
  if (arch_is_in_segment(min_nvidia_gpu_architecture,
                         max_nvidia_gpu_architecture))
    return ext::oneapi::experimental::arch_category::nvidia_gpu;
  if (arch_is_in_segment(min_amd_gpu_architecture, max_amd_gpu_architecture))
    return ext::oneapi::experimental::arch_category::amd_gpu;
  // add "if " when adding new category

  return std::nullopt;
}

template <ext::oneapi::experimental::architecture Arch, typename Compare>
constexpr static bool device_architecture_comparison_aot(Compare comp) {
  constexpr std::optional<ext::oneapi::experimental::arch_category>
      input_arch_category = get_device_architecture_category(Arch);
  constexpr std::optional<ext::oneapi::experimental::architecture>
      current_arch = get_current_architecture_aot();

  if (input_arch_category.has_value() && current_arch.has_value()) {
    std::optional<ext::oneapi::experimental::arch_category>
        current_arch_category = get_device_architecture_category(*current_arch);
    if (current_arch_category.has_value() &&
        (*input_arch_category == *current_arch_category))
      return comp(*current_arch, Arch);
  }
  return false;
}

// Helper object used to implement "else_if_architecture_is",
// "else_if_architecture_is_*" and "otherwise". The "MakeCall" template
// parameter tells whether a previous clause in the "if-elseif-elseif ..." chain
// was true.  When "MakeCall" is false, some previous clause was true, so none
// of the subsequent "else_if_architecture_is", "else_if_architecture_is_*" or
// "otherwise" member functions should call the user's function.
template <bool MakeCall> class if_architecture_helper {
public:
  /// The condition is `true` only if the object F comes from a previous call
  /// whose associated condition is `false` *and* if the device which executes
  /// the `else_if_architecture_is` function has any one of the architectures
  /// listed in the @tparam Archs parameter pack.
  template <ext::oneapi::experimental::architecture... Archs, typename T>
  constexpr auto else_if_architecture_is(T fn) {
    if constexpr (MakeCall && device_architecture_is<Archs...>()) {
      fn();
      return if_architecture_helper<false>{};
    } else {
      (void)fn;
      return if_architecture_helper<MakeCall>{};
    }
  }

  /// The condition is `true` only if the object F comes from a previous call
  /// whose associated condition is `false` *and* if the device which executes
  /// the `else_if_architecture_is` function has an architecture that is in any
  /// one of the categories listed in the @tparam Categories pack.
  template <ext::oneapi::experimental::arch_category... Categories, typename T>
  constexpr auto else_if_architecture_is(T fn) {
    if constexpr (MakeCall &&
                  device_architecture_is_in_categories<Categories...>()) {
      fn();
      return if_architecture_helper<false>{};
    } else {
      (void)fn;
      return if_architecture_helper<MakeCall>{};
    }
  }

  /// The condition is `true` only if the object F comes from a previous call
  /// whose associated condition is `false` *and* if the device which executes
  /// the `else_if_architecture_is_lt` function has an architecture that is in
  /// the same family as @tparam Arch and compares less than @tparam Arch.
  template <ext::oneapi::experimental::architecture Arch, typename T>
  constexpr auto else_if_architecture_is_lt(T fn) {
    if constexpr (MakeCall &&
                  sycl::detail::device_architecture_comparison_aot<Arch>(
                      [](ext::oneapi::experimental::architecture a,
                         ext::oneapi::experimental::architecture b) constexpr {
                        return a < b;
                      })) {
      fn();
      return sycl::detail::if_architecture_helper<false>{};
    } else {
      (void)fn;
      return sycl::detail::if_architecture_helper<MakeCall>{};
    }
  }

  /// The condition is `true` only if the object F comes from a previous call
  /// whose associated condition is `false` *and* if the device which executes
  /// the `else_if_architecture_is_le` function has an architecture that is in
  /// the same family as @tparam Arch and compares less than or equal to @tparam
  /// Arch.
  template <ext::oneapi::experimental::architecture Arch, typename T>
  constexpr auto else_if_architecture_is_le(T fn) {
    if constexpr (MakeCall &&
                  sycl::detail::device_architecture_comparison_aot<Arch>(
                      [](ext::oneapi::experimental::architecture a,
                         ext::oneapi::experimental::architecture b) constexpr {
                        return a <= b;
                      })) {
      fn();
      return sycl::detail::if_architecture_helper<false>{};
    } else {
      (void)fn;
      return sycl::detail::if_architecture_helper<MakeCall>{};
    }
  }

  /// The condition is `true` only if the object F comes from a previous call
  /// whose associated condition is `false` *and* if the device which executes
  /// the `else_if_architecture_is_gt` function has an architecture that is in
  /// the same family as @tparam Arch and compares greater than @tparam Arch.
  template <ext::oneapi::experimental::architecture Arch, typename T>
  constexpr auto else_if_architecture_is_gt(T fn) {
    if constexpr (MakeCall &&
                  sycl::detail::device_architecture_comparison_aot<Arch>(
                      [](ext::oneapi::experimental::architecture a,
                         ext::oneapi::experimental::architecture b) constexpr {
                        return a > b;
                      })) {
      fn();
      return sycl::detail::if_architecture_helper<false>{};
    } else {
      (void)fn;
      return sycl::detail::if_architecture_helper<MakeCall>{};
    }
  }

  /// The condition is `true` only if the object F comes from a previous call
  /// whose associated condition is `false` *and* if the device which executes
  /// the `else_if_architecture_is_ge` function has an architecture that is in
  /// the same family as @tparam Arch and compares greater than or equal to
  /// @tparam Arch.
  template <ext::oneapi::experimental::architecture Arch, typename T>
  constexpr auto else_if_architecture_is_ge(T fn) {
    if constexpr (MakeCall &&
                  sycl::detail::device_architecture_comparison_aot<Arch>(
                      [](ext::oneapi::experimental::architecture a,
                         ext::oneapi::experimental::architecture b) constexpr {
                        return a >= b;
                      })) {
      fn();
      return sycl::detail::if_architecture_helper<false>{};
    } else {
      (void)fn;
      return sycl::detail::if_architecture_helper<MakeCall>{};
    }
  }

  /// The condition is `true` only if the object F comes from a previous call
  /// whose associated condition is `false` *and* if the device which executes
  /// the `else_if_architecture_is_between` function has an architecture that is
  /// in the same family as @tparam Arch1 and is greater than or equal to
  /// @tparam Arch1 and is less than or equal to @tparam Arch2.
  template <ext::oneapi::experimental::architecture Arch1,
            ext::oneapi::experimental::architecture Arch2, typename T>
  constexpr auto else_if_architecture_is_between(T fn) {
    if constexpr (MakeCall &&
                  sycl::detail::device_architecture_comparison_aot<Arch1>(
                      [](ext::oneapi::experimental::architecture a,
                         ext::oneapi::experimental::architecture b) constexpr {
                        return a >= b;
                      }) &&
                  sycl::detail::device_architecture_comparison_aot<Arch2>(
                      [](ext::oneapi::experimental::architecture a,
                         ext::oneapi::experimental::architecture b) constexpr {
                        return a <= b;
                      })) {
      fn();
      return sycl::detail::if_architecture_helper<false>{};
    } else {
      (void)fn;
      return sycl::detail::if_architecture_helper<MakeCall>{};
    }
  }

  template <typename T> constexpr void otherwise(T fn) {
    if constexpr (MakeCall) {
      fn();
    }
  }
};
} // namespace detail

namespace ext::oneapi::experimental {

/// The condition is `true` only if the device which executes the
/// `if_architecture_is` function has any one of the architectures listed in the
/// @tparam Archs pack.
template <architecture... Archs, typename T>
constexpr static auto if_architecture_is(T fn) {
  static_assert(sycl::detail::allowable_aot_mode<Archs...>(),
                "The if_architecture_is function may only be used when AOT "
                "compiling with '-fsycl-targets=spir64_x86_64' or "
                "'-fsycl-targets=*_gpu_*'");
  if constexpr (sycl::detail::device_architecture_is<Archs...>()) {
    fn();
    return sycl::detail::if_architecture_helper<false>{};
  } else {
    (void)fn;
    return sycl::detail::if_architecture_helper<true>{};
  }
}

/// The condition is `true` only if the device which executes the
/// `if_architecture_is` function has an architecture that is in any one of the
/// categories listed in the @tparam Categories pack.
template <arch_category... Categories, typename T>
constexpr static auto if_architecture_is(T fn) {
  if constexpr (sycl::detail::device_architecture_is_in_categories<
                    Categories...>()) {
    fn();
    return sycl::detail::if_architecture_helper<false>{};
  } else {
    (void)fn;
    return sycl::detail::if_architecture_helper<true>{};
  }
}

/// The condition is `true` only if the device which executes the
/// `if_architecture_is_lt` function has an architecture that is in the same
/// family as @tparam Arch and compares less than @tparam Arch.
template <architecture Arch, typename T>
constexpr static auto if_architecture_is_lt(T fn) {
  if constexpr (sycl::detail::device_architecture_comparison_aot<Arch>(
                    [](architecture a, architecture b) constexpr {
                      return a < b;
                    })) {
    fn();
    return sycl::detail::if_architecture_helper<false>{};
  } else {
    (void)fn;
    return sycl::detail::if_architecture_helper<true>{};
  }
}

/// The condition is `true` only if the device which executes the
/// `if_architecture_is_le` function has an architecture that is in the same
/// family as @tparam Arch and compares less than or equal to @tparam Arch.
template <architecture Arch, typename T>
constexpr static auto if_architecture_is_le(T fn) {
  if constexpr (sycl::detail::device_architecture_comparison_aot<Arch>(
                    [](architecture a, architecture b) constexpr {
                      return a <= b;
                    })) {
    fn();
    return sycl::detail::if_architecture_helper<false>{};
  } else {
    (void)fn;
    return sycl::detail::if_architecture_helper<true>{};
  }
}

/// The condition is `true` only if the device which executes the
/// `if_architecture_is_gt` function has an architecture that is in the same
/// family as @tparam Arch and compares greater than @tparam Arch.
template <architecture Arch, typename T>
constexpr static auto if_architecture_is_gt(T fn) {
  if constexpr (sycl::detail::device_architecture_comparison_aot<Arch>(
                    [](architecture a, architecture b) constexpr {
                      return a > b;
                    })) {
    fn();
    return sycl::detail::if_architecture_helper<false>{};
  } else {
    (void)fn;
    return sycl::detail::if_architecture_helper<true>{};
  }
}

/// The condition is `true` only if the device which executes the
/// `if_architecture_is_ge` function has an architecture that is in the same
/// family as @tparam Arch and compares greater than or equal to @tparam Arch.
template <architecture Arch, typename T>
constexpr static auto if_architecture_is_ge(T fn) {
  if constexpr (sycl::detail::device_architecture_comparison_aot<Arch>(
                    [](architecture a, architecture b) constexpr {
                      return a >= b;
                    })) {
    fn();
    return sycl::detail::if_architecture_helper<false>{};
  } else {
    (void)fn;
    return sycl::detail::if_architecture_helper<true>{};
  }
}

/// The condition is `true` only if the device which executes the
/// `if_architecture_is_between` function has an architecture that is in the
/// same family as @tparam Arch1 and is greater than or equal to @tparam
/// Arch1 and is less than or equal to @tparam Arch2.
template <architecture Arch1, architecture Arch2, typename T>
constexpr static auto if_architecture_is_between(T fn) {
  if constexpr (sycl::detail::device_architecture_comparison_aot<Arch1>(
                    [](architecture a, architecture b) constexpr {
                      return a >= b;
                    }) &&
                sycl::detail::device_architecture_comparison_aot<Arch2>(
                    [](architecture a, architecture b) constexpr {
                      return a <= b;
                    })) {
    fn();
    return sycl::detail::if_architecture_helper<false>{};
  } else {
    (void)fn;
    return sycl::detail::if_architecture_helper<true>{};
  }
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
