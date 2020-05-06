//==---------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __SYCL_DEVICE_ONLY__

#ifdef __SYCL_NVPTX__

DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_x();
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_y();
DEVICE_EXTERNAL size_t __spirv_GlobalInvocationId_z();

DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_x();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_y();
DEVICE_EXTERNAL size_t __spirv_LocalInvocationId_z();

#else // __SYCL_NVPTX__

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
extern "C" const size_t_vec __spirv_BuiltInGlobalInvocationId;
extern "C" const size_t_vec __spirv_BuiltInLocalInvocationId;

DEVICE_EXTERNAL inline size_t __spirv_GlobalInvocationId_x() {
  return __spirv_BuiltInGlobalInvocationId.x;
}
DEVICE_EXTERNAL inline size_t __spirv_GlobalInvocationId_y() {
  return __spirv_BuiltInGlobalInvocationId.y;
}
DEVICE_EXTERNAL inline size_t __spirv_GlobalInvocationId_z() {
  return __spirv_BuiltInGlobalInvocationId.z;
}

DEVICE_EXTERNAL inline size_t __spirv_LocalInvocationId_x() {
  return __spirv_BuiltInLocalInvocationId.x;
}
DEVICE_EXTERNAL inline size_t __spirv_LocalInvocationId_y() {
  return __spirv_BuiltInLocalInvocationId.y;
}
DEVICE_EXTERNAL inline size_t __spirv_LocalInvocationId_z() {
  return __spirv_BuiltInLocalInvocationId.z;
}

#endif // __SYCL_NVPTX__

#define DEFINE_FUNC_ID_TO_XYZ_CONVERTER(POSTFIX)                               \
  template <int ID> static inline size_t get##POSTFIX();                       \
  template <> size_t get##POSTFIX<0>() { return __spirv_##POSTFIX##_x(); }     \
  template <> size_t get##POSTFIX<1>() { return __spirv_##POSTFIX##_y(); }     \
  template <> size_t get##POSTFIX<2>() { return __spirv_##POSTFIX##_z(); }

namespace __spirv {

DEFINE_FUNC_ID_TO_XYZ_CONVERTER(GlobalInvocationId);
DEFINE_FUNC_ID_TO_XYZ_CONVERTER(LocalInvocationId);

} // namespace __spirv

#undef DEFINE_FUNC_ID_TO_XYZ_CONVERTER

#define DEFINE_INIT_SIZES(POSTFIX)                                             \
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

namespace __spirv {

DEFINE_INIT_SIZES(GlobalInvocationId)
DEFINE_INIT_SIZES(LocalInvocationId)

} // namespace __spirv

#undef DEFINE_INIT_SIZES

#endif // __SYCL_DEVICE_ONLY__
