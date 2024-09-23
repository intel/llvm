//==----------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#ifdef __SYCL_DEVICE_ONLY__

#include <CL/__spirv/spirv_types.hpp>         // for __ocl_vec_t
#include <sycl/detail/defines_elementary.hpp> // for __DPCPP_SYCL_EXTERNAL

#include <cstddef> // for size_t
#include <cstdint> // for uint8_t

#define __SPIRV_VAR_QUALIFIERS extern "C" const

#if defined(__NVPTX__) || defined(__AMDGCN__) || defined(__SYCL_NATIVE_CPU__)

__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_x();
__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_y();
__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_z();

__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalSize_x();
__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalSize_y();
__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalSize_z();

__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalOffset_x();
__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalOffset_y();
__DPCPP_SYCL_EXTERNAL size_t __spirv_GlobalOffset_z();

__DPCPP_SYCL_EXTERNAL size_t __spirv_NumWorkgroups_x();
__DPCPP_SYCL_EXTERNAL size_t __spirv_NumWorkgroups_y();
__DPCPP_SYCL_EXTERNAL size_t __spirv_NumWorkgroups_z();

__DPCPP_SYCL_EXTERNAL size_t __spirv_WorkgroupSize_x();
__DPCPP_SYCL_EXTERNAL size_t __spirv_WorkgroupSize_y();
__DPCPP_SYCL_EXTERNAL size_t __spirv_WorkgroupSize_z();

__DPCPP_SYCL_EXTERNAL size_t __spirv_WorkgroupId_x();
__DPCPP_SYCL_EXTERNAL size_t __spirv_WorkgroupId_y();
__DPCPP_SYCL_EXTERNAL size_t __spirv_WorkgroupId_z();

__DPCPP_SYCL_EXTERNAL size_t __spirv_LocalInvocationId_x();
__DPCPP_SYCL_EXTERNAL size_t __spirv_LocalInvocationId_y();
__DPCPP_SYCL_EXTERNAL size_t __spirv_LocalInvocationId_z();

__DPCPP_SYCL_EXTERNAL uint32_t __spirv_SubgroupSize();
__DPCPP_SYCL_EXTERNAL uint32_t __spirv_SubgroupMaxSize();
__DPCPP_SYCL_EXTERNAL uint32_t __spirv_NumSubgroups();
__DPCPP_SYCL_EXTERNAL uint32_t __spirv_SubgroupId();
__DPCPP_SYCL_EXTERNAL uint32_t __spirv_SubgroupLocalInvocationId();

#else // defined(__NVPTX__) || defined(__AMDGCN__)

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalSize;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalInvocationId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupSize;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInNumWorkgroups;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInLocalInvocationId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalOffset;

__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupSize;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupMaxSize;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInNumSubgroups;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupId;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupLocalInvocationId;

__SPIRV_VAR_QUALIFIERS __ocl_vec_t<uint32_t, 4> __spirv_BuiltInSubgroupEqMask;
__SPIRV_VAR_QUALIFIERS __ocl_vec_t<uint32_t, 4> __spirv_BuiltInSubgroupGeMask;
__SPIRV_VAR_QUALIFIERS __ocl_vec_t<uint32_t, 4> __spirv_BuiltInSubgroupGtMask;
__SPIRV_VAR_QUALIFIERS __ocl_vec_t<uint32_t, 4> __spirv_BuiltInSubgroupLeMask;
__SPIRV_VAR_QUALIFIERS __ocl_vec_t<uint32_t, 4> __spirv_BuiltInSubgroupLtMask;

__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_x() {
  return __spirv_BuiltInGlobalInvocationId.x;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_y() {
  return __spirv_BuiltInGlobalInvocationId.y;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_z() {
  return __spirv_BuiltInGlobalInvocationId.z;
}

__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalSize_x() {
  return __spirv_BuiltInGlobalSize.x;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalSize_y() {
  return __spirv_BuiltInGlobalSize.y;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalSize_z() {
  return __spirv_BuiltInGlobalSize.z;
}

__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_x() {
  return __spirv_BuiltInGlobalOffset.x;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_y() {
  return __spirv_BuiltInGlobalOffset.y;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_z() {
  return __spirv_BuiltInGlobalOffset.z;
}

__DPCPP_SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_x() {
  return __spirv_BuiltInNumWorkgroups.x;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_y() {
  return __spirv_BuiltInNumWorkgroups.y;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_z() {
  return __spirv_BuiltInNumWorkgroups.z;
}

__DPCPP_SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_x() {
  return __spirv_BuiltInWorkgroupSize.x;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_y() {
  return __spirv_BuiltInWorkgroupSize.y;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_z() {
  return __spirv_BuiltInWorkgroupSize.z;
}

__DPCPP_SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_x() {
  return __spirv_BuiltInWorkgroupId.x;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_y() {
  return __spirv_BuiltInWorkgroupId.y;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_z() {
  return __spirv_BuiltInWorkgroupId.z;
}

__DPCPP_SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_x() {
  return __spirv_BuiltInLocalInvocationId.x;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_y() {
  return __spirv_BuiltInLocalInvocationId.y;
}
__DPCPP_SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_z() {
  return __spirv_BuiltInLocalInvocationId.z;
}

__DPCPP_SYCL_EXTERNAL inline uint32_t __spirv_SubgroupSize() {
  return __spirv_BuiltInSubgroupSize;
}
__DPCPP_SYCL_EXTERNAL inline uint32_t __spirv_SubgroupMaxSize() {
  return __spirv_BuiltInSubgroupMaxSize;
}
__DPCPP_SYCL_EXTERNAL inline uint32_t __spirv_NumSubgroups() {
  return __spirv_BuiltInNumSubgroups;
}
__DPCPP_SYCL_EXTERNAL inline uint32_t __spirv_SubgroupId() {
  return __spirv_BuiltInSubgroupId;
}
__DPCPP_SYCL_EXTERNAL inline uint32_t __spirv_SubgroupLocalInvocationId() {
  return __spirv_BuiltInSubgroupLocalInvocationId;
}

#endif // defined(__NVPTX__) || defined(__AMDGCN__)

#undef __SPIRV_VAR_QUALIFIERS

namespace __spirv {

// Helper function templates to initialize and get vector component from SPIR-V
// built-in variables
#define __SPIRV_DEFINE_INIT_AND_GET_HELPERS(POSTFIX)                           \
  template <int ID> static size_t get##POSTFIX();                              \
  template <> size_t get##POSTFIX<0>() { return __spirv_##POSTFIX##_x(); }     \
  template <> size_t get##POSTFIX<1>() { return __spirv_##POSTFIX##_y(); }     \
  template <> size_t get##POSTFIX<2>() { return __spirv_##POSTFIX##_z(); }     \
                                                                               \
  template <int Dim, class DstT> struct InitSizesST##POSTFIX;                  \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<1, DstT> {                 \
    static DstT initSize() { return {get##POSTFIX<0>()}; }                     \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<2, DstT> {                 \
    static DstT initSize() { return {get##POSTFIX<1>(), get##POSTFIX<0>()}; }  \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<3, DstT> {                 \
    static DstT initSize() {                                                   \
      return {get##POSTFIX<2>(), get##POSTFIX<1>(), get##POSTFIX<0>()};        \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <int Dims, class DstT> DstT init##POSTFIX() {                       \
    return InitSizesST##POSTFIX<Dims, DstT>::initSize();                       \
  }

__SPIRV_DEFINE_INIT_AND_GET_HELPERS(GlobalSize);
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(GlobalInvocationId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(WorkgroupSize)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(NumWorkgroups)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(LocalInvocationId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(WorkgroupId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(GlobalOffset)

#undef __SPIRV_DEFINE_INIT_AND_GET_HELPERS

} // namespace __spirv

#endif // __SYCL_DEVICE_ONLY__
