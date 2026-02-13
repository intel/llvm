//==----------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#ifdef __SYCL_DEVICE_ONLY__

#include <sycl/detail/defines_elementary.hpp> // for __DPCPP_SYCL_EXTERNAL

#include <cstddef> // for size_t
#include <cstdint> // for uint32_t

// SPIR-V built-in variables mapped to function call.

__DPCPP_SYCL_EXTERNAL __attribute__((const)) size_t
__spirv_BuiltInGlobalInvocationId(int);
__DPCPP_SYCL_EXTERNAL __attribute__((const)) size_t
__spirv_BuiltInGlobalSize(int);
__DPCPP_SYCL_EXTERNAL __attribute__((const)) size_t
__spirv_BuiltInGlobalOffset(int);
__DPCPP_SYCL_EXTERNAL __attribute__((const)) size_t
__spirv_BuiltInNumWorkgroups(int);
__DPCPP_SYCL_EXTERNAL __attribute__((const)) size_t
__spirv_BuiltInWorkgroupSize(int);
__DPCPP_SYCL_EXTERNAL __attribute__((const)) size_t
__spirv_BuiltInWorkgroupId(int);
__DPCPP_SYCL_EXTERNAL __attribute__((const)) size_t
__spirv_BuiltInLocalInvocationId(int);

__DPCPP_SYCL_EXTERNAL __attribute__((const)) uint32_t
__spirv_BuiltInSubgroupSize();
__DPCPP_SYCL_EXTERNAL __attribute__((const)) uint32_t
__spirv_BuiltInSubgroupMaxSize();
__DPCPP_SYCL_EXTERNAL __attribute__((const)) uint32_t
__spirv_BuiltInNumSubgroups();
__DPCPP_SYCL_EXTERNAL __attribute__((const)) uint32_t
__spirv_BuiltInSubgroupId();
__DPCPP_SYCL_EXTERNAL __attribute__((const)) uint32_t
__spirv_BuiltInSubgroupLocalInvocationId();

namespace __spirv {

// Helper function templates to initialize and get vector component from SPIR-V
// built-in variables
#define __SPIRV_DEFINE_INIT_AND_GET_HELPERS(POSTFIX)                           \
  template <int ID> size_t get##POSTFIX();                                     \
  template <> size_t get##POSTFIX<0>() { return __spirv_##POSTFIX(0); }        \
  template <> size_t get##POSTFIX<1>() { return __spirv_##POSTFIX(1); }        \
  template <> size_t get##POSTFIX<2>() { return __spirv_##POSTFIX(2); }        \
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

__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInGlobalSize);
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInGlobalInvocationId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInWorkgroupSize)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInNumWorkgroups)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInLocalInvocationId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInWorkgroupId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(BuiltInGlobalOffset)

#undef __SPIRV_DEFINE_INIT_AND_GET_HELPERS

} // namespace __spirv

#endif // __SYCL_DEVICE_ONLY__
