//==---------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#ifdef __SYCL_DEVICE_ONLY__

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));
extern "C" const __attribute__((ocl_constant)) size_t_vec __spirv_BuiltInGlobalSize;
extern "C" const __attribute__((ocl_constant)) size_t_vec __spirv_BuiltInGlobalInvocationId;
extern "C" const __attribute__((ocl_constant)) size_t_vec __spirv_BuiltInWorkgroupSize;
extern "C" const __attribute__((ocl_constant)) size_t_vec __spirv_BuiltInLocalInvocationId;
extern "C" const __attribute__((ocl_constant)) size_t_vec __spirv_BuiltInWorkgroupId;
extern "C" const __attribute__((ocl_constant)) size_t_vec __spirv_BuiltInGlobalOffset;

#define DEFINE_INT_ID_TO_XYZ_CONVERTER(POSTFIX)                                \
  template <int ID> static size_t get##POSTFIX();                              \
  template <> size_t get##POSTFIX<0>() { return __spirv_BuiltIn##POSTFIX.x; }  \
  template <> size_t get##POSTFIX<1>() { return __spirv_BuiltIn##POSTFIX.y; }  \
  template <> size_t get##POSTFIX<2>() { return __spirv_BuiltIn##POSTFIX.z; }

namespace __spirv {

DEFINE_INT_ID_TO_XYZ_CONVERTER(GlobalSize);
DEFINE_INT_ID_TO_XYZ_CONVERTER(GlobalInvocationId)
DEFINE_INT_ID_TO_XYZ_CONVERTER(WorkgroupSize)
DEFINE_INT_ID_TO_XYZ_CONVERTER(LocalInvocationId)
DEFINE_INT_ID_TO_XYZ_CONVERTER(WorkgroupId)
DEFINE_INT_ID_TO_XYZ_CONVERTER(GlobalOffset)

} // namespace __spirv

#undef DEFINE_INT_ID_TO_XYZ_CONVERTER

extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInSubgroupSize;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInSubgroupMaxSize;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInNumSubgroups;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInNumEnqueuedSubgroups;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInSubgroupId;
extern "C" const __attribute__((ocl_constant)) uint32_t __spirv_BuiltInSubgroupLocalInvocationId;

#define DEFINE_INIT_SIZES(POSTFIX)                                             \
                                                                               \
  template <int Dim, class DstT> struct InitSizesST##POSTFIX;                  \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<1, DstT> {                 \
    static void initSize(DstT &Dst) {                                          \
      Dst[0] = get##POSTFIX<0>();                                              \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<2, DstT> {                 \
    static void initSize(DstT &Dst) {                                          \
      Dst[1] = get##POSTFIX<1>();                                              \
      InitSizesST##POSTFIX<1, DstT>::initSize(Dst);                            \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<3, DstT> {                 \
    static void initSize(DstT &Dst) {                                          \
      Dst[2] = get##POSTFIX<2>();                                              \
      InitSizesST##POSTFIX<2, DstT>::initSize(Dst);                            \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <int Dims, class DstT> static void init##POSTFIX(DstT &Dst) {       \
    InitSizesST##POSTFIX<Dims, DstT>::initSize(Dst);                           \
  }

namespace __spirv {

DEFINE_INIT_SIZES(GlobalSize);
DEFINE_INIT_SIZES(GlobalInvocationId)
DEFINE_INIT_SIZES(WorkgroupSize)
DEFINE_INIT_SIZES(LocalInvocationId)
DEFINE_INIT_SIZES(WorkgroupId)
DEFINE_INIT_SIZES(GlobalOffset)

} // namespace __spirv

#undef DEFINE_INIT_SIZES

#endif // __SYCL_DEVICE_ONLY__
