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

#include <CL/sycl/builtins.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/host_util.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>

#include <cstdint>

#define __SEIEED sycl::ext::intel::experimental::esimd::detail

// saturation intrinsics
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_satf(__SEIEED::vector_type_t<T1, SZ> src);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_fptoui_sat(__SEIEED::vector_type_t<T1, SZ> src);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_fptosi_sat(__SEIEED::vector_type_t<T1, SZ> src);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_uutrunc_sat(__SEIEED::vector_type_t<T1, SZ> src);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_ustrunc_sat(__SEIEED::vector_type_t<T1, SZ> src);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_sutrunc_sat(__SEIEED::vector_type_t<T1, SZ> src);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_sstrunc_sat(__SEIEED::vector_type_t<T1, SZ> src);

template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_abs(__SEIEED::vector_type_t<T, SZ> src0);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_ssshl(__SEIEED::vector_type_t<T1, SZ> src0,
              __SEIEED::vector_type_t<T1, SZ> src1);
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_sushl(__SEIEED::vector_type_t<T1, SZ> src0,
              __SEIEED::vector_type_t<T1, SZ> src1);
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_usshl(__SEIEED::vector_type_t<T1, SZ> src0,
              __SEIEED::vector_type_t<T1, SZ> src1);
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_uushl(__SEIEED::vector_type_t<T1, SZ> src0,
              __SEIEED::vector_type_t<T1, SZ> src1);
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_ssshl_sat(__SEIEED::vector_type_t<T1, SZ> src0,
                  __SEIEED::vector_type_t<T1, SZ> src1);
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_sushl_sat(__SEIEED::vector_type_t<T1, SZ> src0,
                  __SEIEED::vector_type_t<T1, SZ> src1);
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_usshl_sat(__SEIEED::vector_type_t<T1, SZ> src0,
                  __SEIEED::vector_type_t<T1, SZ> src1);
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_uushl_sat(__SEIEED::vector_type_t<T1, SZ> src0,
                  __SEIEED::vector_type_t<T1, SZ> src1);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_rol(__SEIEED::vector_type_t<T1, SZ> src0,
            __SEIEED::vector_type_t<T1, SZ> src1);
template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_ror(__SEIEED::vector_type_t<T1, SZ> src0,
            __SEIEED::vector_type_t<T1, SZ> src1);

template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_umulh(__SEIEED::vector_type_t<T, SZ> src0,
              __SEIEED::vector_type_t<T, SZ> src1);
template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_smulh(__SEIEED::vector_type_t<T, SZ> src0,
              __SEIEED::vector_type_t<T, SZ> src1);

template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_frc(__SEIEED::vector_type_t<float, SZ> src0);

/// 3 kinds of max
template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_fmax(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1);
template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_umax(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1);
template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_smax(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1);

template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_lzd(__SEIEED::vector_type_t<T, SZ> src0);

/// 3 kinds of min
template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_fmin(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1);
template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_umin(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1);
template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T, SZ>
__esimd_smin(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1);

template <typename T0, typename T1, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_bfrev(__SEIEED::vector_type_t<T1, SZ> src0);

template <typename T, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<unsigned int, SZ>
__esimd_cbit(__SEIEED::vector_type_t<T, SZ> src0);

template <typename T0, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ> __esimd_bfins(
    __SEIEED::vector_type_t<T0, SZ> src0, __SEIEED::vector_type_t<T0, SZ> src1,
    __SEIEED::vector_type_t<T0, SZ> src2, __SEIEED::vector_type_t<T0, SZ> src3);

template <typename T0, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_bfext(__SEIEED::vector_type_t<T0, SZ> src0,
              __SEIEED::vector_type_t<T0, SZ> src1,
              __SEIEED::vector_type_t<T0, SZ> src2);

template <typename T0, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T0, SZ>
__esimd_fbl(__SEIEED::vector_type_t<T0, SZ> src0);

template <typename T0, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<int, SZ>
__esimd_sfbh(__SEIEED::vector_type_t<T0, SZ> src0);

template <typename T0, int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<uint32_t, SZ>
__esimd_ufbh(__SEIEED::vector_type_t<T0, SZ> src0);

template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_inv(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_log(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_exp(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_sqrt(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_sqrt_ieee(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_rsqrt(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_sin(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_cos(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_pow(__SEIEED::vector_type_t<float, SZ> src0,
            __SEIEED::vector_type_t<float, SZ> src1);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_div_ieee(__SEIEED::vector_type_t<float, SZ> src0,
                 __SEIEED::vector_type_t<float, SZ> src1);

template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_rndd(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_rndu(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_rnde(__SEIEED::vector_type_t<float, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<float, SZ>
__esimd_rndz(__SEIEED::vector_type_t<float, SZ> src0);

template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<double, SZ>
__esimd_sqrt_ieee(__SEIEED::vector_type_t<double, SZ> src0);
template <int SZ>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<double, SZ>
__esimd_div_ieee(__SEIEED::vector_type_t<double, SZ> src0,
                 __SEIEED::vector_type_t<double, SZ> src1);

template <int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION uint32_t
__esimd_pack_mask(__SEIEED::vector_type_t<uint16_t, N> src0);

template <int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<uint16_t, N>
__esimd_unpack_mask(uint32_t src0);

template <typename T1, typename T2, typename T3, typename T4, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T1, N>
__esimd_uudp4a(__SEIEED::vector_type_t<T2, N> src0,
               __SEIEED::vector_type_t<T3, N> src1,
               __SEIEED::vector_type_t<T4, N> src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T1, N>
__esimd_usdp4a(__SEIEED::vector_type_t<T2, N> src0,
               __SEIEED::vector_type_t<T3, N> src1,
               __SEIEED::vector_type_t<T4, N> src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T1, N>
__esimd_sudp4a(__SEIEED::vector_type_t<T2, N> src0,
               __SEIEED::vector_type_t<T3, N> src1,
               __SEIEED::vector_type_t<T4, N> src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T1, N>
__esimd_ssdp4a(__SEIEED::vector_type_t<T2, N> src0,
               __SEIEED::vector_type_t<T3, N> src1,
               __SEIEED::vector_type_t<T4, N> src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T1, N>
__esimd_uudp4a_sat(__SEIEED::vector_type_t<T2, N> src0,
                   __SEIEED::vector_type_t<T3, N> src1,
                   __SEIEED::vector_type_t<T4, N> src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T1, N>
__esimd_usdp4a_sat(__SEIEED::vector_type_t<T2, N> src0,
                   __SEIEED::vector_type_t<T3, N> src1,
                   __SEIEED::vector_type_t<T4, N> src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T1, N>
__esimd_sudp4a_sat(__SEIEED::vector_type_t<T2, N> src0,
                   __SEIEED::vector_type_t<T3, N> src1,
                   __SEIEED::vector_type_t<T4, N> src2);

template <typename T1, typename T2, typename T3, typename T4, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<T1, N>
__esimd_ssdp4a_sat(__SEIEED::vector_type_t<T2, N> src0,
                   __SEIEED::vector_type_t<T3, N> src1,
                   __SEIEED::vector_type_t<T4, N> src2);

// Reduction functions
template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_fmax(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2);

template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_umax(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2);

template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_smax(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2);

template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_fmin(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2);

template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_umin(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2);

template <typename Ty, int N>
__SEIEED::vector_type_t<Ty, N> SYCL_EXTERNAL SYCL_ESIMD_FUNCTION
__esimd_reduced_smin(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2);

template <typename Ty, int N>
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<Ty, N>
__esimd_dp4(__SEIEED::vector_type_t<Ty, N> v1,
            __SEIEED::vector_type_t<Ty, N> v2);

#ifdef __SYCL_DEVICE_ONLY__

// lane-id for reusing scalar math functions.
// Depending upon the SIMT mode(8/16/32), the return value is
// in the range of 0-7, 0-15, or 0-31.
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION int __esimd_lane_id();

// Wrapper for designating a scalar region of code that will be
// vectorized by the backend compiler.
#define __ESIMD_SIMT_BEGIN(N, lane)                                            \
  [&]() SYCL_ESIMD_FUNCTION ESIMD_NOINLINE                                     \
      [[intel::sycl_esimd_vectorize(N)]] {                                     \
    int lane = __esimd_lane_id();
#define __ESIMD_SIMT_END                                                       \
  }                                                                            \
  ();

#define ESIMD_MATH_INTRINSIC_IMPL(type, func)                                  \
  template <int SZ>                                                            \
  SYCL_EXTERNAL SYCL_ESIMD_FUNCTION __SEIEED::vector_type_t<type, SZ>          \
      ocl_##func(__SEIEED::vector_type_t<type, SZ> src0) {                     \
    __SEIEED::vector_type_t<type, SZ> retv;                                    \
    __ESIMD_SIMT_BEGIN(SZ, lane)                                               \
    retv[lane] = sycl::func(src0[lane]);                                       \
    __ESIMD_SIMT_END                                                           \
    return retv;                                                               \
  }

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {
ESIMD_MATH_INTRINSIC_IMPL(float, sin)
ESIMD_MATH_INTRINSIC_IMPL(float, cos)
ESIMD_MATH_INTRINSIC_IMPL(float, exp)
ESIMD_MATH_INTRINSIC_IMPL(float, log)
} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#undef __ESIMD_SIMT_BEGIN
#undef __ESIMD_SIMT_END
#undef ESIMD_MATH_INTRINSIC_IMPL

#else // __SYCL_DEVICE_ONLY__

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

#define __SEIEEED sycl::ext::intel::experimental::esimd::emu::detail

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_satf(__SEIEED::vector_type_t<T1, SZ> src) {
  __SEIEED::vector_type_t<T0, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(src[i], 1);
  }
  return retv;
};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_fptoui_sat(__SEIEED::vector_type_t<T1, SZ> src) {
  __SEIEED::vector_type_t<T0, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(src[i], 1);
  }
  return retv;
};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_fptosi_sat(__SEIEED::vector_type_t<T1, SZ> src) {
  __SEIEED::vector_type_t<T0, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(src[i], 1);
  }
  return retv;
};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_uutrunc_sat(__SEIEED::vector_type_t<T1, SZ> src) {
  __SEIEED::vector_type_t<T0, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(src[i], 1);
  }
  return retv;
};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_ustrunc_sat(__SEIEED::vector_type_t<T1, SZ> src) {
  __SEIEED::vector_type_t<T0, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(src[i], 1);
  }
  return retv;
};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_sutrunc_sat(__SEIEED::vector_type_t<T1, SZ> src) {
  __SEIEED::vector_type_t<T0, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(src[i], 1);
  }
  return retv;
};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_sstrunc_sat(__SEIEED::vector_type_t<T1, SZ> src) {
  __SEIEED::vector_type_t<T0, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(src[i], 1);
  }
  return retv;
};

template <typename T, int SZ>
inline __SEIEED::vector_type_t<T, SZ>
__esimd_abs(__SEIEED::vector_type_t<T, SZ> src0) {
  int i;
  typename __SEIEEED::abstype<T>::type ret;
  __SEIEED::vector_type_t<T, SZ> retv;

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
};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_ssshl(__SEIEED::vector_type_t<T1, SZ> src0,
              __SEIEED::vector_type_t<T1, SZ> src1) {
  int i;
  typename __SEIEEED::maxtype<T1>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0.get(i) << src1.get(i);
    retv[i] = ret;
  }
  return retv;
};
template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_sushl(__SEIEED::vector_type_t<T1, SZ> src0,
              __SEIEED::vector_type_t<T1, SZ> src1) {
  int i;
  typename __SEIEEED::maxtype<T1>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0.get(i) << src1.get(i);
    retv[i] = ret;
  }
  return retv;
};
template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_usshl(__SEIEED::vector_type_t<T1, SZ> src0,
              __SEIEED::vector_type_t<T1, SZ> src1) {
  int i;
  typename __SEIEEED::maxtype<T1>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0.get(i) << src1.get(i);
    retv[i] = ret;
  }
  return retv;
};
template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_uushl(__SEIEED::vector_type_t<T1, SZ> src0,
              __SEIEED::vector_type_t<T1, SZ> src1) {
  int i;
  typename __SEIEEED::maxtype<T1>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0.get(i) << src1.get(i);
    retv[i] = ret;
  }
  return retv;
};
template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_ssshl_sat(__SEIEED::vector_type_t<T1, SZ> src0,
                  __SEIEED::vector_type_t<T1, SZ> src1) {
  int i;
  typename __SEIEEED::maxtype<T1>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0.get(i) << src1.get(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(ret, 1);
  }
  return retv;
};
template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_sushl_sat(__SEIEED::vector_type_t<T1, SZ> src0,
                  __SEIEED::vector_type_t<T1, SZ> src1) {
  int i;
  typename __SEIEEED::maxtype<T1>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0.get(i) << src1.get(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(ret, 1);
  }
  return retv;
};
template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_usshl_sat(__SEIEED::vector_type_t<T1, SZ> src0,
                  __SEIEED::vector_type_t<T1, SZ> src1) {
  int i;
  typename __SEIEEED::maxtype<T1>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0.get(i) << src1.get(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(ret, 1);
  }
  return retv;
};
template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_uushl_sat(__SEIEED::vector_type_t<T1, SZ> src0,
                  __SEIEED::vector_type_t<T1, SZ> src1) {
  int i;
  typename __SEIEEED::maxtype<T1>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0.get(i) << src1.get(i);
    retv[i] = __SEIEEED::satur<T0>::saturate(ret, 1);
  }
  return retv;
};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_rol(__SEIEED::vector_type_t<T1, SZ> src0,
            __SEIEED::vector_type_t<T1, SZ> src1){};

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_ror(__SEIEED::vector_type_t<T1, SZ> src0,
            __SEIEED::vector_type_t<T1, SZ> src1){};

template <typename T, int SZ>
inline __SEIEED::vector_type_t<T, SZ>
__esimd_umulh(__SEIEED::vector_type_t<T, SZ> src0,
              __SEIEED::vector_type_t<T, SZ> src1) {
  int i;
  __SEIEED::vector_type_t<T, SZ> retv;

  for (i = 0; i < SZ; i++) {
    unsigned long long temp;
    SIMDCF_ELEMENT_SKIP(i);
    temp = (long long)src0[i] * (long long)src1[i];
    retv[i] = temp >> 32;
  }
  return retv;
}

template <typename T, int SZ>
inline __SEIEED::vector_type_t<T, SZ>
__esimd_smulh(__SEIEED::vector_type_t<T, SZ> src0,
              __SEIEED::vector_type_t<T, SZ> src1) {
  int i;
  __SEIEED::vector_type_t<T, SZ> retv;

  for (i = 0; i < SZ; i++) {
    long long temp;
    SIMDCF_ELEMENT_SKIP(i);
    temp = (long long)src0[i] * (long long)src1[i];
    retv[i] = temp >> 32;
  }
  return retv;
};

template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_frc(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = src0[i] - floor(src0[i]);
  }
  return retv;
};

/// 3 kinds of max
template <typename T, int SZ>
inline __SEIEED::vector_type_t<T, SZ>
__esimd_fmax(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1) {
  int i;
  __SEIEED::vector_type_t<T, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] >= src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
};
template <typename T, int SZ>
inline __SEIEED::vector_type_t<T, SZ>
__esimd_umax(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1) {
  int i;
  __SEIEED::vector_type_t<T, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] >= src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
};
template <typename T, int SZ>
inline __SEIEED::vector_type_t<T, SZ>
__esimd_smax(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1) {
  int i;
  __SEIEED::vector_type_t<T, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    if (src0[i] >= src1[i]) {
      retv[i] = src0[i];
    } else {
      retv[i] = src1[i];
    }
  }

  return retv;
};

template <typename T, int SZ>
inline __SEIEED::vector_type_t<T, SZ>
__esimd_lzd(__SEIEED::vector_type_t<T, SZ> src0) {
  int i;
  T ret;
  __SEIEED::vector_type_t<T, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    ret = src0[i];
    uint32_t cnt = 0;
    while ((ret & 1u << 31u) == 0 && cnt != 32) {
      cnt++;
      ret = ret << 1;
    }
    retv[i] = cnt;
  }

  return retv;
};

/// 3 kinds of min
template <typename T, int SZ>
inline __SEIEED::vector_type_t<T, SZ>
__esimd_fmin(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1) {
  int i;
  __SEIEED::vector_type_t<T, SZ> retv;

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
inline __SEIEED::vector_type_t<T, SZ>
__esimd_umin(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1) {
  int i;
  __SEIEED::vector_type_t<T, SZ> retv;

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
inline __SEIEED::vector_type_t<T, SZ>
__esimd_smin(__SEIEED::vector_type_t<T, SZ> src0,
             __SEIEED::vector_type_t<T, SZ> src1) {
  int i;
  __SEIEED::vector_type_t<T, SZ> retv;

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

template <typename T0, typename T1, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_bfrev(__SEIEED::vector_type_t<T1, SZ> src0) {
  int i, j;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    T0 input = src0[i];
    T0 output = 0;
    for (j = 0; j < sizeof(T0) * 8; j++) {
      output |= input & 0x1;

      // Don't shift if this was the last one
      if ((j + 1) < (sizeof(T0) * 8)) {
        output <<= 1;
        input >>= 1;
      }
    }
    retv[i] = output;
  }

  return retv;
};

template <typename T, int SZ>
inline __SEIEED::vector_type_t<unsigned int, SZ>
__esimd_cbit(__SEIEED::vector_type_t<T, SZ> src0) {
  int i;
  uint32_t ret;
  __SEIEED::vector_type_t<uint32_t, SZ> retv;

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
};

template <typename T0, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_bfins(__SEIEED::vector_type_t<T0, SZ> width,
              __SEIEED::vector_type_t<T0, SZ> offset,
              __SEIEED::vector_type_t<T0, SZ> val,
              __SEIEED::vector_type_t<T0, SZ> src) {
  int i;
  typename __SEIEEED::maxtype<T0>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    const uint32_t mask = ((1 << width[i]) - 1) << offset[i];
    const uint32_t imask = ~mask;
    ret = (src[i] & imask) | ((val[i] << offset[i] & mask));
    // Sign extend if signed type
    if constexpr (std::is_signed<T0>::value) {
      int m = 1U << (width[i] - 1);
      ret = (ret ^ m) - m;
    }
    retv[i] = ret;
  }

  return retv;
};

template <typename T0, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_bfext(__SEIEED::vector_type_t<T0, SZ> width,
              __SEIEED::vector_type_t<T0, SZ> offset,
              __SEIEED::vector_type_t<T0, SZ> src) {
  int i;
  typename __SEIEEED::maxtype<T0>::type ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

  for (i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    const uint32_t mask = ((1 << width[i]) - 1) << offset[i];
    ret = (src[i] & mask) >> offset[i];
    retv[i] = ret;
  }

  return retv;
};

template <typename T0, int SZ>
inline __SEIEED::vector_type_t<T0, SZ>
__esimd_fbl(__SEIEED::vector_type_t<T0, SZ> src0) {
  int i;
  T0 ret;
  __SEIEED::vector_type_t<T0, SZ> retv;

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
};

template <typename T0, int SZ>
inline __SEIEED::vector_type_t<int, SZ>
__esimd_sfbh(__SEIEED::vector_type_t<T0, SZ> src0) {

  int i, cval;
  int ret;
  __SEIEED::vector_type_t<int, SZ> retv;

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
};

template <typename T0, int SZ>
inline __SEIEED::vector_type_t<uint32_t, SZ>
__esimd_ufbh(__SEIEED::vector_type_t<T0, SZ> src0) {
  uint32_t ret;
  __SEIEED::vector_type_t<uint32_t, SZ> retv;

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
};

template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_inv(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = 1.f / src0[i];
  }
  return retv;
};

template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_log(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = logf(src0[i]) / logf(2.);
  }
  return retv;
};
template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_exp(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = powf(2.f, src0[i]);
  }
  return retv;
};
template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_sqrt(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = sqrt(src0[i]);
  }
  return retv;
};
template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_sqrt_ieee(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = sqrt(src0[i]);
  }
  return retv;
};
template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_rsqrt(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = 1.f / sqrt(src0[i]);
  }
  return retv;
};
template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_sin(__SEIEED::vector_type_t<float, SZ> src) {
  __SEIEED::vector_type_t<float, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = sin(src[i]);
  }
  return retv;
};
template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_cos(__SEIEED::vector_type_t<float, SZ> src) {
  __SEIEED::vector_type_t<float, SZ> retv;
  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = cos(src[i]);
  }
  return retv;
};
template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_pow(__SEIEED::vector_type_t<float, SZ> src0,
            __SEIEED::vector_type_t<float, SZ> src1) {
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = powf(fabs(src0[i]), src1[i]);
  }
  return retv;
};

template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_div_ieee(__SEIEED::vector_type_t<float, SZ> src0,
                 __SEIEED::vector_type_t<float, SZ> src1) {
  __SEIEED::vector_type_t<float, SZ> divinv;
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int idx = 0; idx < SZ; idx += 1) {
    SIMDCF_ELEMENT_SKIP(idx);
    if (src1[idx] == 0.0f) {
      /// Handle Divide-by-zero
      retv[idx] = (src0[idx] < 0) ? (-INFINITY) : INFINITY;
    } else {
      retv[idx] = src0[idx] / src1[idx];
    }
  }

  return retv;
};

template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_rndd(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = floor(src0[i]);
  }
  return retv;
};

template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_rndu(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;
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
};

template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_rnde(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;
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
};

template <int SZ>
inline __SEIEED::vector_type_t<float, SZ>
__esimd_rndz(__SEIEED::vector_type_t<float, SZ> src0) {
  __SEIEED::vector_type_t<float, SZ> retv;
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
};

template <int SZ>
inline __SEIEED::vector_type_t<double, SZ>
__esimd_sqrt_ieee(__SEIEED::vector_type_t<double, SZ> src0) {
  __SEIEED::vector_type_t<double, SZ> retv;

  for (int i = 0; i < SZ; i++) {
    SIMDCF_ELEMENT_SKIP(i);
    retv[i] = sqrt(src0[i]);
  }
  return retv;
};

template <int SZ>
inline __SEIEED::vector_type_t<double, SZ>
__esimd_div_ieee(__SEIEED::vector_type_t<double, SZ> src0,
                 __SEIEED::vector_type_t<double, SZ> src1) {
  __SEIEED::vector_type_t<double, SZ> divinv;
  __SEIEED::vector_type_t<double, SZ> retv;

  for (int idx = 0; idx < SZ; idx += 1) {
    SIMDCF_ELEMENT_SKIP(idx);
    if (src1[idx] == 0.0f) {
      /// Handle Divide-by-zero
      retv[idx] = (src0[idx] < 0) ? (-INFINITY) : INFINITY;
    } else {
      retv[idx] = src0[idx] / src1[idx];
    }
  }

  return retv;
};

template <int N>
inline uint32_t __esimd_pack_mask(__SEIEED::vector_type_t<uint16_t, N> src0) {
  // We don't check the arguments here as this function is only invoked by
  // wrapper code (which does the checks already)
  uint32_t retv = 0;
  for (int i = 0; i < N; i++) {
    if (src0[i] & 0x1) {
      retv |= 0x1 << i;
    }
  }

  return retv;
};

template <int N>
inline __SEIEED::vector_type_t<uint16_t, N> __esimd_unpack_mask(uint32_t src0) {
  __SEIEED::vector_type_t<uint16_t, N> retv;
  for (int i = 0; i < N; i++) {
    if ((src0 >> i) & 0x1) {
      retv[i] = 1;
    } else {
      retv[i] = 0;
    }
  }
  return retv;
};

template <typename T1, typename T2, typename T3, typename T4, int N>
inline __SEIEED::vector_type_t<T1, N>
__esimd_dp4a(__SEIEED::vector_type_t<T2, N> src0,
             __SEIEED::vector_type_t<T3, N> src1,
             __SEIEED::vector_type_t<T4, N> src2) {
  using __SEIEEED::restype_ex;
  typename restype_ex<T2, typename restype_ex<T3, T4>::type>::type reta;
  __SEIEED::vector_type_t<T1, N> retv;

  int src1_a, src1_b, src1_c, src1_d, src2_a, src2_b, src2_c, src2_d, ret;

  uint32_t sat1 =
      __SEIEEED::SetSatur<T2, __SEIEEED::is_inttype<T1>::value>::set() ||
      __SEIEEED::SetSatur<T3, __SEIEEED::is_inttype<T1>::value>::set() ||
      __SEIEEED::SetSatur<T4, __SEIEEED::is_inttype<T1>::value>::set();

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
    retv[i] = __SEIEEED::satur<T1>::saturate(reta, sat1);
  }

  return retv;
};

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_max(__SEIEED::vector_type_t<Ty, N> src1,
                    __SEIEED::vector_type_t<Ty, N> src2) {
  __SEIEED::vector_type_t<Ty, N> retv;
  for (int I = 0; I < N; I++) {
    if (src1[I] >= src2[I]) {
      retv[I] = src1[I];
    } else {
      retv[I] = src2[I];
    }
  }
  return retv;
}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_fmax(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2) {
  return __esimd_reduced_max<Ty, N>(src1, src2);
}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_umax(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2) {
  return __esimd_reduced_max<Ty, N>(src1, src2);
}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_smax(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2) {
  return __esimd_reduced_max<Ty, N>(src1, src2);
}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_min(__SEIEED::vector_type_t<Ty, N> src1,
                    __SEIEED::vector_type_t<Ty, N> src2) {
  __SEIEED::vector_type_t<Ty, N> retv;
  for (int I = 0; I < N; I++) {
    if (src1[I] <= src2[I]) {
      retv[I] = src1[I];
    } else {
      retv[I] = src2[I];
    }
  }
  return retv;
}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_fmin(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2) {
  return __esimd_reduced_min<Ty, N>(src1, src2);
}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_umin(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2) {
  return __esimd_reduced_min<Ty, N>(src1, src2);
}

template <typename Ty, int N>
inline __SEIEED::vector_type_t<Ty, N>
__esimd_reduced_smin(__SEIEED::vector_type_t<Ty, N> src1,
                     __SEIEED::vector_type_t<Ty, N> src2) {
  return __esimd_reduced_min<Ty, N>(src1, src2);
}

#undef __SEIEEED

#endif // #ifdef __SYCL_DEVICE_ONLY__

#undef __SEIEED
