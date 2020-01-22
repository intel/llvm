//==---------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#ifdef __SYCL_DEVICE_ONLY__

#define __SPIRV_VAR_QUALIFIERS extern "C" const __attribute__((opencl_global))

__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupSize;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupMaxSize;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInNumSubgroups;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInNumEnqueuedSubgroups;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupId;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupLocalInvocationId;

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalSize;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalInvocationId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupSize;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInNumWorkgroups;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInLocalInvocationId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalOffset;

#undef __SPIRV_VAR_QUALIFIERS

namespace __spirv {

// Helper function templates to initialize and get vector component from SPIR-V
// built-in variables
#define __SPIRV_DEFINE_INIT_AND_GET_HELPERS(POSTFIX)                           \
  template <int ID> static size_t get##POSTFIX();                              \
  template <> size_t get##POSTFIX<0>() { return __spirv_BuiltIn##POSTFIX.x; }  \
  template <> size_t get##POSTFIX<1>() { return __spirv_BuiltIn##POSTFIX.y; }  \
  template <> size_t get##POSTFIX<2>() { return __spirv_BuiltIn##POSTFIX.z; }  \
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
  template <int Dims, class DstT> static DstT init##POSTFIX() {                \
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
