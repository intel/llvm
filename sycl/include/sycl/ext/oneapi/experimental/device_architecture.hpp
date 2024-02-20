//===- device_architecture.hpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class architecture {
  // If new element is added to this enum:
  //
  // Update
  //   - sycl_ext_oneapi_device_architecture specification doc
  //   - "-fsycl-targets" description in sycl/doc/UsersManual.md
  //
  // Add
  //   - __SYCL_TARGET_<ARCH>__ to the compiler driver and to all places below
  //   - the unique ID of the new architecture in SYCL RT source code to support
  //     querying the device architecture
  //
  x86_64,
  intel_cpu_spr,
  intel_cpu_gnr,
  intel_gpu_bdw,
  intel_gpu_skl,
  intel_gpu_kbl,
  intel_gpu_cfl,
  intel_gpu_apl,
  intel_gpu_bxt = intel_gpu_apl,
  intel_gpu_glk,
  intel_gpu_whl,
  intel_gpu_aml,
  intel_gpu_cml,
  intel_gpu_icllp,
  intel_gpu_ehl,
  intel_gpu_jsl = intel_gpu_ehl,
  intel_gpu_tgllp,
  intel_gpu_rkl,
  intel_gpu_adl_s,
  intel_gpu_rpl_s = intel_gpu_adl_s,
  intel_gpu_adl_p,
  intel_gpu_adl_n,
  intel_gpu_dg1,
  intel_gpu_acm_g10,
  intel_gpu_dg2_g10 = intel_gpu_acm_g10,
  intel_gpu_acm_g11,
  intel_gpu_dg2_g11 = intel_gpu_acm_g11,
  intel_gpu_acm_g12,
  intel_gpu_dg2_g12 = intel_gpu_acm_g12,
  intel_gpu_pvc,
  intel_gpu_pvc_vg,
  // NVIDIA architectures
  nvidia_gpu_sm_50,
  nvidia_gpu_sm_52,
  nvidia_gpu_sm_53,
  nvidia_gpu_sm_60,
  nvidia_gpu_sm_61,
  nvidia_gpu_sm_62,
  nvidia_gpu_sm_70,
  nvidia_gpu_sm_72,
  nvidia_gpu_sm_75,
  nvidia_gpu_sm_80,
  nvidia_gpu_sm_86,
  nvidia_gpu_sm_87,
  nvidia_gpu_sm_89,
  nvidia_gpu_sm_90,
  // AMD architectures
  amd_gpu_gfx700,
  amd_gpu_gfx701,
  amd_gpu_gfx702,
  amd_gpu_gfx801,
  amd_gpu_gfx802,
  amd_gpu_gfx803,
  amd_gpu_gfx805,
  amd_gpu_gfx810,
  amd_gpu_gfx900,
  amd_gpu_gfx902,
  amd_gpu_gfx904,
  amd_gpu_gfx906,
  amd_gpu_gfx908,
  amd_gpu_gfx909,
  amd_gpu_gfx90a,
  amd_gpu_gfx90c,
  amd_gpu_gfx940,
  amd_gpu_gfx941,
  amd_gpu_gfx942,
  amd_gpu_gfx1010,
  amd_gpu_gfx1011,
  amd_gpu_gfx1012,
  amd_gpu_gfx1013,
  amd_gpu_gfx1030,
  amd_gpu_gfx1031,
  amd_gpu_gfx1032,
  amd_gpu_gfx1033,
  amd_gpu_gfx1034,
  amd_gpu_gfx1035,
  amd_gpu_gfx1036,
  amd_gpu_gfx1100,
  amd_gpu_gfx1101,
  amd_gpu_gfx1102,
  amd_gpu_gfx1103,
  amd_gpu_gfx1150,
  amd_gpu_gfx1151,
  amd_gpu_gfx1200,
  amd_gpu_gfx1201,
  // Update "detail::max_architecture" below if you add new elements here!
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

} // namespace ext::oneapi::experimental

namespace detail {

static constexpr ext::oneapi::experimental::architecture max_architecture =
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

struct IsAOTForArchitectureClass {
  // Allocate an array of size == size of
  // ext::oneapi::experimental::architecture enum.
  bool arr[static_cast<int>(max_architecture) + 1];

  using arch = ext::oneapi::experimental::architecture;

  constexpr IsAOTForArchitectureClass() : arr() {
    arr[static_cast<int>(arch::x86_64)] = __SYCL_TARGET_INTEL_X86_64__ == 1;
    arr[static_cast<int>(arch::intel_gpu_bdw)] =
        __SYCL_TARGET_INTEL_GPU_BDW__ == 1;
    arr[static_cast<int>(arch::intel_gpu_skl)] =
        __SYCL_TARGET_INTEL_GPU_SKL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_kbl)] =
        __SYCL_TARGET_INTEL_GPU_KBL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_cfl)] =
        __SYCL_TARGET_INTEL_GPU_CFL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_apl)] =
        __SYCL_TARGET_INTEL_GPU_APL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_glk)] =
        __SYCL_TARGET_INTEL_GPU_GLK__ == 1;
    arr[static_cast<int>(arch::intel_gpu_whl)] =
        __SYCL_TARGET_INTEL_GPU_WHL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_aml)] =
        __SYCL_TARGET_INTEL_GPU_AML__ == 1;
    arr[static_cast<int>(arch::intel_gpu_cml)] =
        __SYCL_TARGET_INTEL_GPU_CML__ == 1;
    arr[static_cast<int>(arch::intel_gpu_icllp)] =
        __SYCL_TARGET_INTEL_GPU_ICLLP__ == 1;
    arr[static_cast<int>(arch::intel_gpu_ehl)] =
        __SYCL_TARGET_INTEL_GPU_EHL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_tgllp)] =
        __SYCL_TARGET_INTEL_GPU_TGLLP__ == 1;
    arr[static_cast<int>(arch::intel_gpu_rkl)] =
        __SYCL_TARGET_INTEL_GPU_RKL__ == 1;
    arr[static_cast<int>(arch::intel_gpu_adl_s)] =
        __SYCL_TARGET_INTEL_GPU_ADL_S__ == 1;
    arr[static_cast<int>(arch::intel_gpu_adl_p)] =
        __SYCL_TARGET_INTEL_GPU_ADL_P__ == 1;
    arr[static_cast<int>(arch::intel_gpu_adl_n)] =
        __SYCL_TARGET_INTEL_GPU_ADL_N__ == 1;
    arr[static_cast<int>(arch::intel_gpu_dg1)] =
        __SYCL_TARGET_INTEL_GPU_DG1__ == 1;
    arr[static_cast<int>(arch::intel_gpu_acm_g10)] =
        __SYCL_TARGET_INTEL_GPU_ACM_G10__ == 1;
    arr[static_cast<int>(arch::intel_gpu_acm_g11)] =
        __SYCL_TARGET_INTEL_GPU_ACM_G11__ == 1;
    arr[static_cast<int>(arch::intel_gpu_acm_g12)] =
        __SYCL_TARGET_INTEL_GPU_ACM_G12__ == 1;
    arr[static_cast<int>(arch::intel_gpu_pvc)] =
        __SYCL_TARGET_INTEL_GPU_PVC__ == 1;
    arr[static_cast<int>(arch::intel_gpu_pvc_vg)] =
        __SYCL_TARGET_INTEL_GPU_PVC_VG__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_50)] =
        __SYCL_TARGET_NVIDIA_GPU_SM50__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_52)] =
        __SYCL_TARGET_NVIDIA_GPU_SM52__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_53)] =
        __SYCL_TARGET_NVIDIA_GPU_SM53__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_60)] =
        __SYCL_TARGET_NVIDIA_GPU_SM60__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_61)] =
        __SYCL_TARGET_NVIDIA_GPU_SM61__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_62)] =
        __SYCL_TARGET_NVIDIA_GPU_SM62__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_70)] =
        __SYCL_TARGET_NVIDIA_GPU_SM70__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_72)] =
        __SYCL_TARGET_NVIDIA_GPU_SM72__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_75)] =
        __SYCL_TARGET_NVIDIA_GPU_SM75__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_80)] =
        __SYCL_TARGET_NVIDIA_GPU_SM80__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_86)] =
        __SYCL_TARGET_NVIDIA_GPU_SM86__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_87)] =
        __SYCL_TARGET_NVIDIA_GPU_SM87__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_89)] =
        __SYCL_TARGET_NVIDIA_GPU_SM89__ == 1;
    arr[static_cast<int>(arch::nvidia_gpu_sm_90)] =
        __SYCL_TARGET_NVIDIA_GPU_SM90__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx700)] =
        __SYCL_TARGET_AMD_GPU_GFX700__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx701)] =
        __SYCL_TARGET_AMD_GPU_GFX701__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx702)] =
        __SYCL_TARGET_AMD_GPU_GFX702__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx801)] =
        __SYCL_TARGET_AMD_GPU_GFX801__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx802)] =
        __SYCL_TARGET_AMD_GPU_GFX802__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx803)] =
        __SYCL_TARGET_AMD_GPU_GFX803__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx805)] =
        __SYCL_TARGET_AMD_GPU_GFX805__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx810)] =
        __SYCL_TARGET_AMD_GPU_GFX810__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx900)] =
        __SYCL_TARGET_AMD_GPU_GFX900__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx902)] =
        __SYCL_TARGET_AMD_GPU_GFX902__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx904)] =
        __SYCL_TARGET_AMD_GPU_GFX904__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx906)] =
        __SYCL_TARGET_AMD_GPU_GFX906__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx908)] =
        __SYCL_TARGET_AMD_GPU_GFX908__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx909)] =
        __SYCL_TARGET_AMD_GPU_GFX909__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx90a)] =
        __SYCL_TARGET_AMD_GPU_GFX90A__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx90c)] =
        __SYCL_TARGET_AMD_GPU_GFX90C__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx940)] =
        __SYCL_TARGET_AMD_GPU_GFX940__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx941)] =
        __SYCL_TARGET_AMD_GPU_GFX941__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx942)] =
        __SYCL_TARGET_AMD_GPU_GFX942__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1010)] =
        __SYCL_TARGET_AMD_GPU_GFX1010__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1011)] =
        __SYCL_TARGET_AMD_GPU_GFX1011__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1012)] =
        __SYCL_TARGET_AMD_GPU_GFX1012__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1030)] =
        __SYCL_TARGET_AMD_GPU_GFX1030__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1031)] =
        __SYCL_TARGET_AMD_GPU_GFX1031__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1032)] =
        __SYCL_TARGET_AMD_GPU_GFX1032__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1033)] =
        __SYCL_TARGET_AMD_GPU_GFX1033__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1034)] =
        __SYCL_TARGET_AMD_GPU_GFX1034__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1035)] =
        __SYCL_TARGET_AMD_GPU_GFX1035__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1036)] =
        __SYCL_TARGET_AMD_GPU_GFX1036__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1100)] =
        __SYCL_TARGET_AMD_GPU_GFX1100__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1101)] =
        __SYCL_TARGET_AMD_GPU_GFX1101__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1102)] =
        __SYCL_TARGET_AMD_GPU_GFX1102__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1103)] =
        __SYCL_TARGET_AMD_GPU_GFX1103__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1150)] =
        __SYCL_TARGET_AMD_GPU_GFX1150__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1151)] =
        __SYCL_TARGET_AMD_GPU_GFX1151__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1200)] =
        __SYCL_TARGET_AMD_GPU_GFX1200__ == 1;
    arr[static_cast<int>(arch::amd_gpu_gfx1201)] =
        __SYCL_TARGET_AMD_GPU_GFX1201__ == 1;
  }
};

// One entry for each enumerator in "architecture" telling whether the AOT
// target matches that architecture.
static constexpr IsAOTForArchitectureClass is_aot_for_architecture;

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
  return (is_aot_for_architecture.arr[static_cast<int>(Archs)] || ...);
}

// Helper object used to implement "else_if_architecture_is" and "otherwise".
// The "MakeCall" template parameter tells whether a previous clause in the
// "if-elseif-elseif ..." chain was true.  When "MakeCall" is false, some
// previous clause was true, so none of the subsequent
// "else_if_architecture_is" or "otherwise" member functions should call the
// user's function.
template <bool MakeCall> class if_architecture_helper {
public:
  template <ext::oneapi::experimental::architecture... Archs, typename T>
  constexpr auto else_if_architecture_is(T fnTrue) {
    if constexpr (MakeCall && device_architecture_is<Archs...>()) {
      fnTrue();
      return if_architecture_helper<false>{};
    } else {
      (void)fnTrue;
      return if_architecture_helper<MakeCall>{};
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

template <architecture... Archs, typename T>
constexpr static auto if_architecture_is(T fnTrue) {
  static_assert(sycl::detail::allowable_aot_mode<Archs...>(),
                "The if_architecture_is function may only be used when AOT "
                "compiling with '-fsycl-targets=spir64_x86_64' or "
                "'-fsycl-targets=*_gpu_*'");
  if constexpr (sycl::detail::device_architecture_is<Archs...>()) {
    fnTrue();
    return sycl::detail::if_architecture_helper<false>{};
  } else {
    (void)fnTrue;
    return sycl::detail::if_architecture_helper<true>{};
  }
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
