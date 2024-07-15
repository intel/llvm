//===- device_architecture.hpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint> // for uint64_t
#include <optional>
#include <utility> // for std::integer_sequence

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class architecture : uint64_t {
#define __SYCL_ARCHITECTURE(NAME, VAL) NAME = VAL,
#define __SYCL_ARCHITECTURE_ALIAS(NAME, VAL) NAME = VAL,
#include <sycl/ext/oneapi/experimental/architectures.def>
#undef __SYCL_ARCHITECTURE
#undef __SYCL_ARCHITECTURE_ALIAS
};

enum class arch_category {
  // If new element is added to this enum:
  //
  // Add
  //   - "detail::min_<new_category>_architecture" variable below
  //   - "detail::max_<new_category>_architecture" variable below
  //
  // Update
  //   - "detail::get_category_min_architecture()" function below
  //   - "detail::get_category_max_architecture()" function below
  //   - "detail::get_device_architecture_category()" function below
  //   - sycl_ext_oneapi_device_architecture specification doc
  //
  intel_gpu = 0,
  nvidia_gpu = 1,
  amd_gpu = 2,
  // TODO: add intel_cpu = 3,
};

} // namespace ext::oneapi::experimental

namespace detail {

static constexpr ext::oneapi::experimental::architecture
    min_intel_gpu_architecture =
        ext::oneapi::experimental::architecture::intel_gpu_bdw;
static constexpr ext::oneapi::experimental::architecture
    max_intel_gpu_architecture =
        ext::oneapi::experimental::architecture::intel_gpu_lnl_m;

static constexpr ext::oneapi::experimental::architecture
    min_nvidia_gpu_architecture =
        ext::oneapi::experimental::architecture::nvidia_gpu_sm_50;
static constexpr ext::oneapi::experimental::architecture
    max_nvidia_gpu_architecture =
        ext::oneapi::experimental::architecture::nvidia_gpu_sm_90a;

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
#ifndef __SYCL_TARGET_INTEL_GPU_MTL_U__
#define __SYCL_TARGET_INTEL_GPU_MTL_U__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_MTL_H__
#define __SYCL_TARGET_INTEL_GPU_MTL_H__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_ARL_H__
#define __SYCL_TARGET_INTEL_GPU_ARL_H__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_BMG_G21__
#define __SYCL_TARGET_INTEL_GPU_BMG_G21__ 0
#endif
#ifndef __SYCL_TARGET_INTEL_GPU_LNL_M__
#define __SYCL_TARGET_INTEL_GPU_LNL_M__ 0
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
    (__SYCL_TARGET_INTEL_GPU_MTL_U__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_MTL_H__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_ARL_H__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_BMG_G21__ == 1) ||
    (__SYCL_TARGET_INTEL_GPU_LNL_M__ == 1) ||
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
#if __SYCL_TARGET_INTEL_GPU_MTL_U__
  return ext::oneapi::experimental::architecture::intel_gpu_mtl_u;
#endif
#if __SYCL_TARGET_INTEL_GPU_MTL_H__
  return ext::oneapi::experimental::architecture::intel_gpu_mtl_h;
#endif
#if __SYCL_TARGET_INTEL_GPU_ARL_H__
  return ext::oneapi::experimental::architecture::intel_gpu_arl_h;
#endif
#if __SYCL_TARGET_INTEL_GPU_BMG_G21__
  return ext::oneapi::experimental::architecture::intel_gpu_bmg_g21;
#endif
#if __SYCL_TARGET_INTEL_GPU_LNL_M__
  return ext::oneapi::experimental::architecture::intel_gpu_lnl_m;
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

constexpr auto device_arch_compare_op_lt =
    [](ext::oneapi::experimental::architecture a,
       ext::oneapi::experimental::architecture b) constexpr { return a < b; };
constexpr auto device_arch_compare_op_le =
    [](ext::oneapi::experimental::architecture a,
       ext::oneapi::experimental::architecture b) constexpr { return a <= b; };
constexpr auto device_arch_compare_op_gt =
    [](ext::oneapi::experimental::architecture a,
       ext::oneapi::experimental::architecture b) constexpr { return a > b; };
constexpr auto device_arch_compare_op_ge =
    [](ext::oneapi::experimental::architecture a,
       ext::oneapi::experimental::architecture b) constexpr { return a >= b; };

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
                      device_arch_compare_op_lt)) {
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
                      device_arch_compare_op_le)) {
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
                      device_arch_compare_op_gt)) {
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
                      device_arch_compare_op_ge)) {
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
                      device_arch_compare_op_ge) &&
                  sycl::detail::device_architecture_comparison_aot<Arch2>(
                      device_arch_compare_op_le)) {
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

namespace detail {
// Call the callable object "fn" only when this code runs on a device which
// has a certain set of aspects or a particular architecture.
//
// Condition is a parameter pack of int's that define a simple expression
// language which tells the set of aspects or architectures that the device
// must have in order to enable the call.  See the "Condition*" values below.
template <typename T, typename... Condition>
#ifdef __SYCL_DEVICE_ONLY__
[[__sycl_detail__::add_ir_attributes_function(
    "sycl-call-if-on-device-conditionally", true)]]
#endif
void call_if_on_device_conditionally(T fn, Condition...) {
  fn();
}

// The "Condition" parameter pack above is a sequence of int's that define an
// expression tree.  Each node represents a boolean subexpression:
//
// ConditionAspect -       Next int is a value from "enum aspect".  The
//                           subexpression is true if the device has this
//                           aspect.
// ConditionArchitecture - Next int is a value from "enum architecture".  The
//                           subexpression is true if the device has this
//                           architecture.
// ConditionNot -          Next int is the root of another subexpression S1.
//                           This subexpression is true if S1 is false.
// ConditionAnd -          Next int is the root of another subexpression S1.
//                           The int following that subexpression is the root
//                           of another subexpression S2.  This subexpression
//                           is true if both S1 and S2 are true.
// ConditionOr -           Next int is the root of another subexpression S1.
//                           The int following that subexpression is the root
//                           of another subexpression S2.  This subexpression
//                           is true if either S1 or S2 are true.
//
// These values are stored in the application's executable, so they are
// effectively part of the ABI.  Therefore, any change to an existing value
// is an ABI break.
//
// There is no programmatic reason for the values to be negative.  They are
// negative only by convention to make it easier for humans to distinguish them
// from aspect or architecture values (which are positive).
static constexpr int ConditionAspect = -1;
static constexpr int ConditionArchitecture = -2;
static constexpr int ConditionNot = -3;
static constexpr int ConditionAnd = -4;
static constexpr int ConditionOr = -5;

// Metaprogramming helper to construct a ConditionOr expression for a sequence
// of architectures.  "ConditionAnyArchitectureBuilder<Archs...>::seq" is an
// "std::integer_sequence" representing the expression.
template <architecture... Archs> struct ConditionAnyArchitectureBuilder;

template <architecture Arch, architecture... Archs>
struct ConditionAnyArchitectureBuilder<Arch, Archs...> {
  template <int I1, int I2, int I3, int... Is>
  static auto append(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, I1, I2, I3, Is...>{};
  }
  using rest = typename ConditionAnyArchitectureBuilder<Archs...>::seq;
  static constexpr int arch = static_cast<int>(Arch);
  using seq =
      decltype(append<ConditionOr, ConditionArchitecture, arch>(rest{}));
};

template <architecture Arch> struct ConditionAnyArchitectureBuilder<Arch> {
  static constexpr int arch = static_cast<int>(Arch);
  using seq = std::integer_sequence<int, ConditionArchitecture, arch>;
};

// Metaprogramming helper to construct a ConditionNot expression.
// ConditionNotBuilder<Exp>::seq" is an "std::integer_sequence" representing
// the expression.
template <typename Exp> struct ConditionNotBuilder {
  template <int I, int... Is>
  static auto append(std::integer_sequence<int, Is...>) {
    return std::integer_sequence<int, I, Is...>{};
  }
  using rest = typename Exp::seq;
  using seq = decltype(append<ConditionNot>(rest{}));
};

// Metaprogramming helper to construct a ConditionAnd expression.
// "ConditionAndBuilder<Exp1, Exp2>::seq" is an "std::integer_sequence"
// representing the expression.
template <typename Exp1, typename Exp2> struct ConditionAndBuilder {
  template <int I, int... I1s, int... I2s>
  static auto append(std::integer_sequence<int, I1s...>,
                     std::integer_sequence<int, I2s...>) {
    return std::integer_sequence<int, I, I1s..., I2s...>{};
  }
  using rest1 = typename Exp1::seq;
  using rest2 = typename Exp2::seq;
  using seq = decltype(append<ConditionAnd>(rest1{}, rest2{}));
};

// Metaprogramming helper to construct a ConditionOr expression.
// "ConditionOrBuilder<Exp1, Exp2>::seq" is an "std::integer_sequence"
// representing the expression.
template <typename Exp1, typename Exp2> struct ConditionOrBuilder {
  template <int I, int... I1s, int... I2s>
  static auto append(std::integer_sequence<int, I1s...>,
                     std::integer_sequence<int, I2s...>) {
    return std::integer_sequence<int, I, I1s..., I2s...>{};
  }
  using rest1 = typename Exp1::seq;
  using rest2 = typename Exp2::seq;
  using seq = decltype(append<ConditionOr>(rest1{}, rest2{}));
};

// Helper function to call call_if_on_device_conditionally() while converting
// the "std::integer_sequence" for a condition expression into individual
// arguments of type int.
template <typename T, int... Is>
void call_if_on_device_conditionally_helper(T fn,
                                            std::integer_sequence<int, Is...>) {
  call_if_on_device_conditionally(fn, Is...);
}

// Same sort of helper object for "else_if_architecture_is".
template <typename MakeCallIf> class if_architecture_is_helper {
public:
  template <architecture... Archs, typename T,
            typename = std::enable_if<std::is_invocable_v<T>>>
  auto else_if_architecture_is(T fn) {
    using make_call_if =
        ConditionAndBuilder<MakeCallIf,
                            ConditionAnyArchitectureBuilder<Archs...>>;
    using make_else_call_if = ConditionAndBuilder<
        MakeCallIf,
        ConditionNotBuilder<ConditionAnyArchitectureBuilder<Archs...>>>;

    using cond = typename make_call_if::seq;
    call_if_on_device_conditionally_helper(fn, cond{});
    return if_architecture_is_helper<make_else_call_if>{};
  }

  template <typename T> void otherwise(T fn) {
    using cond = typename MakeCallIf::seq;
    call_if_on_device_conditionally_helper(fn, cond{});
  }
};

} // namespace detail

#ifdef SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE_NEW_DESIGN_IMPL
template <architecture... Archs, typename T>
static auto if_architecture_is(T fn) {
  using make_call_if = detail::ConditionAnyArchitectureBuilder<Archs...>;
  using make_else_call_if = detail::ConditionNotBuilder<make_call_if>;

  using cond = typename make_call_if::seq;
  detail::call_if_on_device_conditionally_helper(fn, cond{});
  return detail::if_architecture_is_helper<make_else_call_if>{};
}
#else
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
#endif // SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE_NEW_DESIGN_IMPL

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
                    sycl::detail::device_arch_compare_op_lt)) {
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
                    sycl::detail::device_arch_compare_op_le)) {
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
                    sycl::detail::device_arch_compare_op_gt)) {
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
                    sycl::detail::device_arch_compare_op_ge)) {
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
                    sycl::detail::device_arch_compare_op_ge) &&
                sycl::detail::device_architecture_comparison_aot<Arch2>(
                    sycl::detail::device_arch_compare_op_le)) {
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
