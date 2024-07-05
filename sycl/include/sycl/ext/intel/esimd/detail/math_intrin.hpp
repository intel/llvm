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
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/detail/util.hpp>

#include <cstdint>

#define __ESIMD_raw_vec_t(T, SZ)                                               \
  __ESIMD_DNS::vector_type_t<__ESIMD_DNS::__raw_t<T>, SZ>
#define __ESIMD_cpp_vec_t(T, SZ)                                               \
  __ESIMD_DNS::vector_type_t<__ESIMD_DNS::__cpp_t<T>, SZ>

// The following spirv intrinsics declarations are put here to avoid unintended
// use by other targets where it causes run time failures due to the fact that
// they are implemented for INTEL GPU only.
template <typename T> extern __DPCPP_SYCL_EXTERNAL T __spirv_ocl_native_exp2(T);
template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_native_exp2(__ESIMD_raw_vec_t(T, N));

template <typename T>
extern __DPCPP_SYCL_EXTERNAL T __spirv_ocl_native_recip(T);
template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_native_recip(__ESIMD_raw_vec_t(T, N));

template <typename T> extern __DPCPP_SYCL_EXTERNAL T __spirv_ocl_native_cos(T);
template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_native_cos(__ESIMD_raw_vec_t(T, N));

template <typename T> extern __DPCPP_SYCL_EXTERNAL T __spirv_ocl_native_log2(T);
template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_native_log2(__ESIMD_raw_vec_t(T, N));

template <typename T>
extern __DPCPP_SYCL_EXTERNAL T __spirv_ocl_native_rsqrt(T);
template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_native_rsqrt(__ESIMD_raw_vec_t(T, N));

template <typename T> extern __DPCPP_SYCL_EXTERNAL T __spirv_ocl_native_sin(T);
template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_native_sin(__ESIMD_raw_vec_t(T, N));

template <typename T> extern __DPCPP_SYCL_EXTERNAL T __spirv_ocl_native_sqrt(T);
template <typename T, int N>
extern __DPCPP_SYCL_EXTERNAL __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_native_sqrt(__ESIMD_raw_vec_t(T, N));

template <typename T>
extern __DPCPP_SYCL_EXTERNAL T __spirv_ocl_native_powr(T, T);
template <typename T, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_native_powr(__ESIMD_raw_vec_t(T, N), __ESIMD_raw_vec_t(T, N));

template <typename T, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_fabs(__ESIMD_raw_vec_t(T, N)) __ESIMD_INTRIN_END;

template <typename T, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_s_abs(__ESIMD_raw_vec_t(T, N)) __ESIMD_INTRIN_END;

template <typename T, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, N)
    __spirv_ocl_fmin(__ESIMD_raw_vec_t(T, N),
                     __ESIMD_raw_vec_t(T, N)) __ESIMD_INTRIN_END;

// saturation intrinsics
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sat(__ESIMD_raw_vec_t(T1, SZ) src) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_fptoui_sat(__ESIMD_raw_vec_t(T1, SZ) src) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_fptosi_sat(__ESIMD_raw_vec_t(T1, SZ) src) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uutrunc_sat(__ESIMD_raw_vec_t(T1, SZ) src) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ustrunc_sat(__ESIMD_raw_vec_t(T1, SZ) src) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sutrunc_sat(__ESIMD_raw_vec_t(T1, SZ) src) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sstrunc_sat(__ESIMD_raw_vec_t(T1, SZ) src) __ESIMD_INTRIN_END;

/// 3 kinds of max
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_fmax(__ESIMD_raw_vec_t(T, SZ) src0,
                 __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umax(__ESIMD_raw_vec_t(T, SZ) src0,
                 __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smax(__ESIMD_raw_vec_t(T, SZ) src0,
                 __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;

/// 3 kinds of min, the missing fmin uses spir-v instrinsics above
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_umin(__ESIMD_raw_vec_t(T, SZ) src0,
                 __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;
template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_smin(__ESIMD_raw_vec_t(T, SZ) src0,
                 __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;

template <typename T, int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<unsigned int, SZ>
    __esimd_cbit(__ESIMD_raw_vec_t(T, SZ) src0) __ESIMD_INTRIN_END;

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_fbl(__ESIMD_raw_vec_t(T0, SZ) src0) __ESIMD_INTRIN_END;

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(int, SZ)
    __esimd_sfbh(__ESIMD_raw_vec_t(T0, SZ) src0) __ESIMD_INTRIN_END;

template <typename T0, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(uint32_t, SZ)
    __esimd_ufbh(__ESIMD_raw_vec_t(T0, SZ) src0) __ESIMD_INTRIN_END;

#define __ESIMD_UNARY_EXT_MATH_INTRIN(name)                                    \
  template <class T, int SZ>                                                   \
  __ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)                                      \
      __esimd_##name(__ESIMD_raw_vec_t(T, SZ) src) __ESIMD_INTRIN_END

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
    __esimd_pow(__ESIMD_raw_vec_t(T, SZ) src0,
                __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;

template <class T, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T, SZ)
    __esimd_ieee_div(__ESIMD_raw_vec_t(T, SZ) src0,
                     __ESIMD_raw_vec_t(T, SZ) src1) __ESIMD_INTRIN_END;

template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndd(__ESIMD_DNS::vector_type_t<float, SZ> src0) __ESIMD_INTRIN_END;
template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndu(__ESIMD_DNS::vector_type_t<float, SZ> src0) __ESIMD_INTRIN_END;
template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rnde(__ESIMD_DNS::vector_type_t<float, SZ> src0) __ESIMD_INTRIN_END;
template <int SZ>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<float, SZ>
__esimd_rndz(__ESIMD_DNS::vector_type_t<float, SZ> src0) __ESIMD_INTRIN_END;

template <int N>
__ESIMD_INTRIN uint32_t __esimd_pack_mask(
    __ESIMD_DNS::vector_type_t<uint16_t, N> src0) __ESIMD_INTRIN_END;

template <int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<uint16_t, N>
__esimd_unpack_mask(uint32_t src0) __ESIMD_INTRIN_END;

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_uudp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                   __ESIMD_raw_vec_t(T4, N) src2) __ESIMD_INTRIN_END;

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_usdp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                   __ESIMD_raw_vec_t(T4, N) src2) __ESIMD_INTRIN_END;

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_sudp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                   __ESIMD_raw_vec_t(T4, N) src2) __ESIMD_INTRIN_END;

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_ssdp4a(__ESIMD_raw_vec_t(T2, N) src0, __ESIMD_raw_vec_t(T3, N) src1,
                   __ESIMD_raw_vec_t(T4, N) src2) __ESIMD_INTRIN_END;

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_uudp4a_sat(__ESIMD_raw_vec_t(T2, N) src0,
                       __ESIMD_raw_vec_t(T3, N) src1,
                       __ESIMD_raw_vec_t(T4, N) src2) __ESIMD_INTRIN_END;

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_usdp4a_sat(__ESIMD_raw_vec_t(T2, N) src0,
                       __ESIMD_raw_vec_t(T3, N) src1,
                       __ESIMD_raw_vec_t(T4, N) src2) __ESIMD_INTRIN_END;

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_sudp4a_sat(__ESIMD_raw_vec_t(T2, N) src0,
                       __ESIMD_raw_vec_t(T3, N) src1,
                       __ESIMD_raw_vec_t(T4, N) src2) __ESIMD_INTRIN_END;

template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T1, N)
    __esimd_ssdp4a_sat(__ESIMD_raw_vec_t(T2, N) src0,
                       __ESIMD_raw_vec_t(T3, N) src1,
                       __ESIMD_raw_vec_t(T4, N) src2) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ssshl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sushl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_usshl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uushl(__ESIMD_raw_vec_t(T1, SZ) src0,
                  __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ssshl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_sushl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_usshl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_uushl_sat(__ESIMD_raw_vec_t(T1, SZ) src0,
                      __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;

template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_rol(__ESIMD_raw_vec_t(T1, SZ) src0,
                __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;
template <typename T0, typename T1, int SZ>
__ESIMD_INTRIN __ESIMD_raw_vec_t(T0, SZ)
    __esimd_ror(__ESIMD_raw_vec_t(T1, SZ) src0,
                __ESIMD_raw_vec_t(T1, SZ) src1) __ESIMD_INTRIN_END;

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
inline namespace _V1 {
namespace ext::intel::esimd::detail {
// TODO support half vectors in std sycl math functions.
ESIMD_MATH_INTRINSIC_IMPL(float, sin)
ESIMD_MATH_INTRINSIC_IMPL(float, cos)
ESIMD_MATH_INTRINSIC_IMPL(float, exp)
ESIMD_MATH_INTRINSIC_IMPL(float, log)
} // namespace ext::intel::esimd::detail
} // namespace _V1
} // namespace sycl

#undef __ESIMD_SIMT_BEGIN
#undef __ESIMD_SIMT_END
#undef ESIMD_MATH_INTRINSIC_IMPL

#endif // #ifdef __SYCL_DEVICE_ONLY__

#undef __ESIMD_raw_vec_t
#undef __ESIMD_cpp_vec_t

/// @endcond ESIMD_DETAIL
