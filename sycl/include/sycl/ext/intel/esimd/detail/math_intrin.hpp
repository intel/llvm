//==------------ math_intrin.hpp - DPC++ Explicit SIMD API -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares Explicit SIMD math intrinsics used to implement working with
// the SIMD classes objects.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

#include <sycl/builtins.hpp>
#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/elem_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/host_util.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/detail/util.hpp>

#include <cstdint>

#define __ESIMD_raw_vec_t(T, SZ)                                               \
  __ESIMD_DNS::vector_type_t<__ESIMD_DNS::__raw_t<T>, SZ>
#define __ESIMD_cpp_vec_t(T, SZ)                                               \
  __ESIMD_DNS::vector_type_t<__ESIMD_DNS::__cpp_t<T>, SZ>

// saturation intrinsics
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sat(__ESIMD_raw_vec_t(T1, SZ) src);

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_fptoui_sat(__ESIMD_raw_vec_t(T1, SZ) src);

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_fptosi_sat(__ESIMD_raw_vec_t(T1, SZ) src);

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uutrunc_sat(__ESIMD_raw_vec_t(T1, SZ) src);

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ustrunc_sat(__ESIMD_raw_vec_t(T1, SZ) src);

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sutrunc_sat(__ESIMD_raw_vec_t(T1, SZ) src);

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sstrunc_sat(__ESIMD_raw_vec_t(T1, SZ) src);

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_abs(__ESIMD_raw_vec_t(T, SZ) src0);

/// 3 kinds of max
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_fmax(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umax(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smax(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);

/// 3 kinds of min
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_fmin(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umin(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smin(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<unsigned int, SZ>
    __esimd_cbit(__ESIMD_raw_vec_t(T, SZ) src0);

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_fbl(__ESIMD_raw_vec_t(T0, SZ) src0);

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(int, SZ)
    __esimd_sfbh(__ESIMD_raw_vec_t(T0, SZ) src0);

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(uint32_t, SZ)
    __esimd_ufbh(__ESIMD_raw_vec_t(T0, SZ) src0);

#define __ESIMD_UNARY_EXT_MATH_INTRIN(name)                                    \
  template <class T, int SZ>                                                   \
  __ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)                                      \
      __esimd_##name(__ESIMD_raw_vec_t(T, SZ) src)

__ESIMD_UNARY_EXT_MATH_INTRIN(inv);
__ESIMD_UNARY_EXT_MATH_INTRIN(log);
__ESIMD_UNARY_EXT_MATH_INTRIN(exp);
__ESIMD_UNARY_EXT_MATH_INTRIN(sqrt);
__ESIMD_UNARY_EXT_MATH_INTRIN(ieee_sqrt);
__ESIMD_UNARY_EXT_MATH_INTRIN(rsqrt);
__ESIMD_UNARY_EXT_MATH_INTRIN(sin);
__ESIMD_UNARY_EXT_MATH_INTRIN(cos);

#undef __ESIMD_UNARY_EXT_MATH_INTRIN

template <class T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_pow(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1);

template <class T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_ieee_div(__ESIMD_raw_vec_t(T, SZ) src0,
                     __ESIMD_raw_vec_t(T, SZ) src1);

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndd(__ESIMD_DNS::vector_type_t<float, SZ> src0);
template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndu(__ESIMD_DNS::vector_type_t<float, SZ> src0);
template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rnde(__ESIMD_DNS::vector_type_t<float, SZ> src0);
template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndz(__ESIMD_DNS::vector_type_t<float, SZ> src0);

template <int N>
__ESIMD_INTRIN uint32_t
__esimd_pack_mask(__ESIMD_DNS::vector_type_t<uint16_t, N> src0);

template <int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<uint16_t, N>
__esimd_unpack_mask(uint32_t src0);

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_uudp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                   __ESIMD_raw_vec_t(T4, N) src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_usdp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                   __ESIMD_raw_vec_t(T4, N) src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_sudp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                   __ESIMD_raw_vec_t(T4, N) src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_ssdp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                   __ESIMD_raw_vec_t(T4, N) src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_uudp4a_sat(__ESIMD_raw_vec_t(T2, N) src0,
                       __ESIMD_raw_vec_t(T3, N) src1,
                       __ESIMD_raw_vec_t(T4, N) src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_usdp4a_sat(__ESIMD_raw_vec_t(T2, N) src0,
                       __ESIMD_raw_vec_t(T3, N) src1,
                       __ESIMD_raw_vec_t(T4, N) src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_sudp4a_sat(__ESIMD_raw_vec_t(T2, N) src0,
                       __ESIMD_raw_vec_t(T3, N) src1,
                       __ESIMD_raw_vec_t(T4, N) src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_ssdp4a_sat(__ESIMD_raw_vec_t(T2, N) src0,
                       __ESIMD_raw_vec_t(T3, N) src1,
                       __ESIMD_raw_vec_t(T4, N) src2);

#ifdef __SYCL_DEVICE_ONLY__

// lane-id for reusing scalar math functions.
// Depending upon the SIMT mode(8/16/32), the return value is
// in the range of 0-7, 0-15, or 0-31.
__ESIMD_INTRIN int __esimd_lane_id();

// Wrapper for designating a scalar region of code that will be
// vectorized by the backend compiler.
#define __ESIMD_SIMT_BEGIN(N, lane)                                            \
  [&]() SYCL_ESIMD_FUNCTION ESIMD_NOINLINE [[intel::sycl_esimd_vectorize(N)]] {                                     \
    int lane = __esimd_lane_id();
#define __ESIMD_SIMT_END                                                       \
  }                                                                            \
  ();

#define ESIMD_MATH_INTRINSIC_IMPL(type, func)                                  \
  template <int SZ>                                                            \
  __ESIMD_INTRIN __ESIMD_raw_vec_t(type, SZ)                                   \
      ocl_##func(__ESIMD_raw_vec_t(type, SZ) src0) {                           \
    __ESIMD_raw_vec_t(type, SZ) retv;                                          \
    __ESIMD_SIMT_BEGIN(SZ, lane)                                               \
    retv[lane] = sycl::func(src0[lane]);                                       \
    __ESIMD_SIMT_END                                                           \
    return retv;                                                               \
  }

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::esimd::detail {
// TODO support half vectors in std sycl math functions.
ESIMD_MATH_INTRINSIC_IMPL(float, sin)
ESIMD_MATH_INTRINSIC_IMPL(float, cos)
ESIMD_MATH_INTRINSIC_IMPL(float, exp)
ESIMD_MATH_INTRINSIC_IMPL(float, log)
} // namespace ext::intel::esimd::detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef __ESIMD_SIMT_BEGIN
#undef __ESIMD_SIMT_END
#undef ESIMD_MATH_INTRINSIC_IMPL

#else // __SYCL_DEVICE_ONLY__

// Typical implementation of a generic intrinsic supporting non-standard
// types (half, bfloat*,...) should be like this:
// - user type information is encoded in template parameters, but function
//   parameters and return type are raw types
// - before use, parameters are converted to EnclosingCppT
// - return value is calculated using the converted parameters,
//   but before return it is converted back to the user type and is bitcast
//   (that's what .data() basically does) to the raw type
//
// template <class T, int SZ>
// __ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ) __esimd_intrin(
//   __ESIMD_raw_vec_t(T, SZ) raw_src0, __ESIMD_raw_vec_t(T, SZ) raw_src1) {
//
//   simd<T, SZ> ret;
//   simd<T, SZ> src0{raw_src0};
//   simd<T, SZ> src1{raw_src1};
//   ret = function_of(src0, src1);
//   return ret.data();
//
// TODO Not following this approach in some of the intrinsics, and performing
// calculations on the raw type will lead to runtime compuation error. A guard
//   if (__ESIMD_DNS::is_wrapper_elem_type_v<T>) __ESIMD_UNSUPPORTED_ON_HOST;
// is temporarily used for now, until wrapper types are supported by these
// intrinsics.

template <typename T>
inline T extract(const uint32_t &width, const uint32_t &offset, uint32_t src,
                 const uint32_t &sign_extend) {
  uint32_t mask = ((1 << width) - 1) << offset;
  T ret = (src & mask) >> offset;
  if (sign_extend) {
    if ((src >> (offset + width - 1)) & 0x1) {
      uint32_t sign_extend = ((1 << (32 - width)) - 1) << width;
      ret = ret | sign_extend;
    }
  }

  return ret;
}

#define __ESIMD_DEFAULT_HOST_SATURATE_INTRIN(name)                             \
  template <typename T0, typename T1, int SZ>                                  \
  __ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)                                     \
      __esimd_##name(__ESIMD_raw_vec_t(T1, SZ) src) {                          \
    __ESIMD_raw_vec_t(T0, SZ) retv;                                            \
    for (int i = 0; i < SZ; i++) {                                             \
      SIMDCF_ELEMENT_SKIP(i);                                                  \
      retv[i] = __ESIMD_EMU_DNS::satur<T0>::template saturate<T1>(src[i], 1);  \
    }                                                                          \
    return retv;                                                               \
  }

__ESIMD_DEFAULT_HOST_SATURATE_INTRIN(sat)
__ESIMD_DEFAULT_HOST_SATURATE_INTRIN(fptoui_sat)
__ESIMD_DEFAULT_HOST_SATURATE_INTRIN(fptosi_sat)
__ESIMD_DEFAULT_HOST_SATURATE_INTRIN(uutrunc_sat)
__ESIMD_DEFAULT_HOST_SATURATE_INTRIN(ustrunc_sat)
__ESIMD_DEFAULT_HOST_SATURATE_INTRIN(sutrunc_sat)
__ESIMD_DEFAULT_HOST_SATURATE_INTRIN(sstrunc_sat)

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_abs(__ESIMD_raw_vec_t(T, SZ) src0) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  typename __ESIMD_EMU_DNS::abstype<T>::type ret;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] < 0) {
      ret = -(src0[i]);
    } else {
      ret = (src0[i]);
    }
    retv[i] = ret;
  }
  return retv;
}

/// 3 kinds of max
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_fmax(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] >= src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umax(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] >= src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smax(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] >= src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
}

/// 3 kinds of min
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_fmin(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] < src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
};

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umin(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] < src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smin(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] < src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<unsigned int, SZ>
__esimd_cbit(__ESIMD_raw_vec_t(T, SZ) src0) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  uint32_t ret;
  __ESIMD_raw_vec_t(uint32_t, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i];
    uint32_t cnt = 0;
    for (int j = 0; j < sizeof(T) * 8; j++) {
      if ((ret & 1u) == 1) {
        cnt++;
      }
      ret = ret >> 1;
    }
    retv[i] = cnt;
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_fbl(__ESIMD_raw_vec_t(T, SZ) src0) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i;
  T ret;
  __ESIMD_raw_vec_t(T, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i];
    uint32_t cnt = 0;
    while ((ret & 1u) == 0 && cnt != 32) {
      cnt++;
      ret = ret >> 1;
    }
    if (src0[i] == 0x0) {
      retv[i] = 0xFFFFFFFF;
    } else {
      retv[i] = cnt;
    }
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(int, SZ)
    __esimd_sfbh(__ESIMD_raw_vec_t(T, SZ) src0) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  int i, cval;
  int ret;
  __ESIMD_raw_vec_t(int, SZ) retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i];
    uint32_t cnt = 0;
    if (((ret >> 31u) & 1u) == 1) {
      cval = 1;
    } else {
      cval = 0;
    }
    while (((ret >> 31u) & 1u) == cval && cnt != 32) {
      cnt++;
      ret = ret << 1;
    }

    if ((src0[i] == 0xFFFFFFFF) || (src0[i] == 0x00000000)) {
      retv[i] = 0xFFFFFFFF;
    } else {
      retv[i] = cnt;
    }
  }

  return retv;
}

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(uint32_t, SZ)
    __esimd_ufbh(__ESIMD_raw_vec_t(T, SZ) src0) {
  if (__ESIMD_DNS::is_wrapper_elem_type_v<T>)
    __ESIMD_UNSUPPORTED_ON_HOST;
  uint32_t ret;
  __ESIMD_raw_vec_t(uint32_t, SZ) retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i];
    uint32_t cnt = 0;
    while ((ret & (1u << 31u)) == 0 && cnt != 32) {
      cnt++;
      ret = ret << 1;
    }
    if (src0[i] == 0x00000000) {
      retv[i] = 0xFFFFFFFF;
    } else {
      retv[i] = cnt;
    }
  }

  return retv;
}

// Host intrinsics are implemented via converting elements to enclosing Cpp
// type (always 'float' except ieee_sqrt, which can be 'double'), applying
// standard C++ library math function and converting back to the element type.
//
#define __ESIMD_UNARY_EXT_MATH_HOST_INTRIN(name, formula)                      \
  template <class T, int SZ>                                                   \
  __ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)                                      \
      __esimd_##name(__ESIMD_raw_vec_t(T, SZ) src) {                           \
    using CppT = __ESIMD_DNS::__cpp_t<T>;                                      \
    using CppVecT = __ESIMD_cpp_vec_t(T, SZ);                                  \
    CppVecT ret_cpp{0};                                                        \
    CppVecT src_cpp = __ESIMD_DNS::convert_vector<CppT, T, SZ>(src);           \
                                                                               \
    for (int i = 0; i < SZ; i++) {                                             \
      SIMDCF_ELEMENT_SKIP(i);                                                  \
      ret_cpp[i] = formula;                                                    \
    }                                                                          \
    __ESIMD_raw_vec_t(T, SZ) ret =                                             \
        __ESIMD_DNS::convert_vector<T, CppT, SZ>(ret_cpp);                     \
    return ret;                                                                \
  }

__ESIMD_UNARY_EXT_MATH_HOST_INTRIN(inv, 1.f / src_cpp[i])
__ESIMD_UNARY_EXT_MATH_HOST_INTRIN(log, logf(src_cpp[i]) / logf(2.f))
__ESIMD_UNARY_EXT_MATH_HOST_INTRIN(exp, powf(2.f, src_cpp[i]))
__ESIMD_UNARY_EXT_MATH_HOST_INTRIN(sqrt, sqrt(src_cpp[i]))
__ESIMD_UNARY_EXT_MATH_HOST_INTRIN(ieee_sqrt, sqrt(src_cpp[i]))
__ESIMD_UNARY_EXT_MATH_HOST_INTRIN(rsqrt, 1.f / sqrt(src_cpp[i]))
__ESIMD_UNARY_EXT_MATH_HOST_INTRIN(sin, sin(src_cpp[i]))
__ESIMD_UNARY_EXT_MATH_HOST_INTRIN(cos, cos(src_cpp[i]))

#undef __ESIMD_UNARY_EXT_MATH_HOST_INTRIN

template <class T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_pow(__ESIMD_raw_vec_t(T, SZ) src0, __ESIMD_raw_vec_t(T, SZ) src1) {
  using CppT = __ESIMD_DNS::__cpp_t<T>;
  using CppVecT = __ESIMD_cpp_vec_t(T, SZ);

  CppVecT cpp_src0 = __ESIMD_DNS::convert_vector<CppT, T, SZ>(src0);
  CppVecT cpp_src1 = __ESIMD_DNS::convert_vector<CppT, T, SZ>(src1);
  CppVecT cpp_res;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    cpp_res[i] = std::pow(std::fabs(cpp_src0[i]), cpp_src1[i]);
  }
  return __ESIMD_DNS::convert_vector<T, CppT, SZ>(cpp_res);
}

template <class T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_ieee_div(__ESIMD_raw_vec_t(T, SZ) src0,
                     __ESIMD_raw_vec_t(T, SZ) src1) {
  using CppT = __ESIMD_DNS::__cpp_t<T>;
  using CppVecT = __ESIMD_cpp_vec_t(T, SZ);

  CppVecT cpp_src0 = __ESIMD_DNS::convert_vector<CppT, T, SZ>(src0);
  CppVecT cpp_src1 = __ESIMD_DNS::convert_vector<CppT, T, SZ>(src1);
  CppVecT cpp_res;

  for (int i = 0; i < SZ; i += 1) {
    SIMDCF_ELEMENT_SKIP(i);
    if (cpp_src1[i] == 0) {
      /// Handle Divide-by-zero
      cpp_res[i] = (cpp_src0[i] < 0) ? (-INFINITY) : INFINITY;
    } else {
      cpp_res[i] = cpp_src0[i] / cpp_src1[i];
    }
  }
  return __ESIMD_DNS::convert_vector<T, CppT, SZ>(cpp_res);
}

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndd(__ESIMD_DNS::vector_type_t<float, SZ> src0) {
  __ESIMD_DNS::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = floor(src0[i]);
  }
  return retv;
}

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndu(__ESIMD_DNS::vector_type_t<float, SZ> src0) {
  __ESIMD_DNS::vector_type_t<float, SZ> retv;
  int increment;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] - floor(src0[i]) > 0.0f) {
      increment = 1;
    } else {
      increment = 0;
    }

    retv[i] = floor(src0[i]) + increment;
  }

  return retv;
}

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rnde(__ESIMD_DNS::vector_type_t<float, SZ> src0) {
  __ESIMD_DNS::vector_type_t<float, SZ> retv;
  int increment;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] - floor(src0[i]) > 0.5f) {
      increment = 1;
    } else if (src0[i] - floor(src0[i]) < 0.5f) {
      increment = 0;
    } else {
      increment = (int(floor(src0[i])) % 2 == 1);
    }

    retv[i] = floor(src0[i]) + increment;
  }

  return retv;
}

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndz(__ESIMD_DNS::vector_type_t<float, SZ> src0) {
  __ESIMD_DNS::vector_type_t<float, SZ> retv;
  int increment;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (fabs(src0[i]) < fabs(floor(src0[i]))) {
      increment = 1;
    } else {
      increment = 0;
    }
    retv[i] = floor(src0[i]) + increment;
  }

  return retv;
}

template <int N>
__ESIMD_INTRIN uint32_t
__esimd_pack_mask(__ESIMD_DNS::vector_type_t<uint16_t, N> src0) {
  // We don't check the arguments here as this function is only invoked by
  // wrapper code (which does the checks already)
  uint32_t retv = 0;
  for (int i = 0; i < N; i++) {
    if (src0[i] != 0) {
      retv |= 0x1 << i;
    }
  }

  return retv;
}

template <int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<uint16_t, N>
__esimd_unpack_mask(uint32_t src0) {
  __ESIMD_DNS::vector_type_t<uint16_t, N> retv = 0;
  for (int i = 0; i < N; i++) {
    if ((src0 >> i) & 0x1) {
      retv[i] = 1;
    }
  }
  return retv;
}

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_dp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                 __ESIMD_raw_vec_t(T4, N) src2) {
#define __ESIMD_WR(T) __ESIMD_DNS::is_wrapper_elem_type_v<T>
  if (__ESIMD_WR(T1) || __ESIMD_WR(T2) || __ESIMD_WR(T3) || __ESIMD_WR(T4))
    __ESIMD_UNSUPPORTED_ON_HOST;
#undef __ESIMD_IS_WR
  using __ESIMD_EMU_DNS::restype_ex;
  typename restype_ex<T2, typename restype_ex<T3, T4>::type>::type reta;
  __ESIMD_raw_vec_t(T1, N) retv;

  int src1_a, src1_b, src1_c, src1_d, src2_a, src2_b, src2_c, src2_d, ret;

  uint32_t sat1 =
      __ESIMD_EMU_DNS::SetSatur<
          T2, __ESIMD_EMU_DNS::is_inttype<T1>::value>::set() ||
      __ESIMD_EMU_DNS::SetSatur<
          T3, __ESIMD_EMU_DNS::is_inttype<T1>::value>::set() ||
      __ESIMD_EMU_DNS::SetSatur<T4,
                                __ESIMD_EMU_DNS::is_inttype<T1>::value>::set();

  for (uint32_t i = 0; i < N; i++) {

    SIMDCF_ELEMENT_SKIP(i);

    src1_a = extract<short>(8, 0, src1[i], 0);
    src1_b = extract<short>(8, 8, src1[i], 0);
    src1_c = extract<short>(8, 16, src1[i], 0);
    src1_d = extract<short>(8, 24, src1[i], 0);
    src2_a = extract<short>(8, 0, src2[i], 0);
    src2_b = extract<short>(8, 8, src2[i], 0);
    src2_c = extract<short>(8, 16, src2[i], 0);
    src2_d = extract<short>(8, 24, src2[i], 0);

    ret = src1_a * src2_a + src1_b * src2_b + src1_c * src2_c + src1_d * src2_d;
    reta = ret + src0[i];
    retv[i] = __ESIMD_EMU_DNS::satur<T1>::template saturate(reta, sat1);
  }

  return retv;
}

#endif // #ifdef __SYCL_DEVICE_ONLY__

#undef __ESIMD_raw_vec_t
#undef __ESIMD_cpp_vec_t

/// @endcond ESIMD_DETAIL
