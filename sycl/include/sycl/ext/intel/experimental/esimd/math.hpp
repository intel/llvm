//==-------------- math.hpp - DPC++ Explicit SIMD API   --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement experimental Explicit SIMD math APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/math_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental::esimd {

/// @addtogroup sycl_esimd_bitmanip
/// @{

/// Count the number of 1-bits.
/// @tparam T element type.
/// @tparam N vector length.
/// @return the popcounted vector.
template <typename T, int N>
__ESIMD_API std::enable_if_t<std::is_integral_v<T> && sizeof(T) < 8,
                             __ESIMD_NS::simd<T, N>>
popcount(__ESIMD_NS::simd<T, N> vec) {
  return __spirv_ocl_popcount<T, N>(vec.data());
}

/// Count the number of leading zeros.
/// If the input is 0, the number of total bits is returned.
/// @tparam T element type.
/// @tparam N vector length.
/// @return vector with number of leading zeros of the input vector.
template <typename T, int N>
__ESIMD_API std::enable_if_t<std::is_integral_v<T> && sizeof(T) < 8,
                             __ESIMD_NS::simd<T, N>>
clz(__ESIMD_NS::simd<T, N> vec) {
  return __spirv_ocl_clz<T, N>(vec.data());
}

/// Count the number of trailing zeros.
/// @tparam T element type.
/// @tparam N vector length.
/// @return vector with number of trailing zeros of the input vector.
template <typename T, int N>
__ESIMD_API std::enable_if_t<std::is_integral_v<T> && sizeof(T) < 8,
                             __ESIMD_NS::simd<T, N>>
ctz(__ESIMD_NS::simd<T, N> vec) {
  return __spirv_ocl_ctz<T, N>(vec.data());
}

/// @} sycl_esimd_bitmanip

/// @addtogroup sycl_esimd_math
/// @{

/// Computes the 64-bit result of two 32-bit element vectors \p src0 and
/// \p src1 multiplication. The result is returned in two separate 32-bit
/// vectors. The low 32-bit parts of the results are written to the output
/// parameter \p rmd and the upper parts of the results are returned from
/// the function.
template <typename T, typename T0, typename T1, int N>
__ESIMD_API __ESIMD_NS::simd<T, N> imul_impl(__ESIMD_NS::simd<T, N> &rmd,
                                             __ESIMD_NS::simd<T0, N> src0,
                                             __ESIMD_NS::simd<T1, N> src1) {
  static_assert(__ESIMD_DNS::is_dword_type<T>::value &&
                    __ESIMD_DNS::is_dword_type<T0>::value &&
                    __ESIMD_DNS::is_dword_type<T1>::value,
                "expected 32-bit integer vector operands.");
  using Comp32T = __ESIMD_DNS::computation_type_t<T0, T1>;
  auto Src0 = src0.template bit_cast_view<Comp32T>();
  auto Src1 = src1.template bit_cast_view<Comp32T>();

  // Compute the result using 64-bit multiplication operation.
  using Comp64T =
      std::conditional_t<std::is_signed_v<Comp32T>, int64_t, uint64_t>;
  __ESIMD_NS::simd<Comp64T, N> Product64 = Src0;
  Product64 *= Src1;

  // Split the 32-bit high and low parts to return them from this function.
  auto Product32 = Product64.template bit_cast_view<T>();
  if constexpr (N == 1) {
    rmd = Product32[0];
    return Product32[1];
  } else {
    rmd = Product32.template select<N, 2>(0);
    return Product32.template select<N, 2>(1);
  }
}

/// Computes the 64-bit multiply result of two 32-bit integer vectors \p src0
/// and \p src1. The result is returned in two separate 32-bit vectors.
/// The low 32-bit parts of the result are written to the output parameter
/// \p rmd and the upper parts of the result are returned from the function.
template <typename T, typename T0, typename T1, int N>
__ESIMD_API __ESIMD_NS::simd<T, N> imul(__ESIMD_NS::simd<T, N> &rmd,
                                        __ESIMD_NS::simd<T0, N> src0,
                                        __ESIMD_NS::simd<T1, N> src1) {
  return imul_impl<T, T0, T1, N>(rmd, src0, src1);
}

/// Computes the 64-bit multiply result of 32-bit integer vector \p src0 and
/// 32-bit integer scalar \p src1. The result is returned in two separate 32-bit
/// vectors. The low 32-bit parts of the result is written to the output
/// parameter \p rmd and the upper part of the results is returned from
/// the function.
template <typename T, typename T0, typename T1, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_dword_type<T1>::value,
                             __ESIMD_NS::simd<T, N>>
imul(__ESIMD_NS::simd<T, N> &rmd, __ESIMD_NS::simd<T0, N> src0, T1 src1) {
  __ESIMD_NS::simd<T1, N> Src1V = src1;
  return esimd::imul_impl<T, T0, T1, N>(rmd, src0, Src1V);
}

/// Computes the 64-bit multiply result of a scalar 32-bit integer \p src0 and
/// 32-bit integer vector \p src1. The result is returned in two separate 32-bit
/// vectors. The low 32-bit parts of the result is written to the output
/// parameter \p rmd and the upper part of the results is returned from
/// the function.
template <typename T, typename T0, typename T1, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_dword_type<T0>::value,
                             __ESIMD_NS::simd<T, N>>
imul(__ESIMD_NS::simd<T, N> &rmd, T0 src0, __ESIMD_NS::simd<T1, N> src1) {
  __ESIMD_NS::simd<T0, N> Src0V = src0;
  return esimd::imul_impl<T, T0, T1, N>(rmd, Src0V, src1);
}

/// Computes the 64-bit multiply result of two scalar 32-bit integer values
/// \p src0 and \p src1. The result is returned in two separate 32-bit scalars.
/// The low 32-bit part of the result is written to the output parameter \p rmd
/// and the upper part of the result is returned from the function.
template <typename T, typename T0, typename T1>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_dword_type<T>::value &&
                                 __ESIMD_DNS::is_dword_type<T0>::value &&
                                 __ESIMD_DNS::is_dword_type<T1>::value,
                             T>
imul(T &rmd, T0 src0, T1 src1) {
  __ESIMD_NS::simd<T, 1> RmdV = rmd;
  __ESIMD_NS::simd<T0, 1> Src0V = src0;
  __ESIMD_NS::simd<T1, 1> Src1V = src1;
  __ESIMD_NS::simd<T, 1> Res =
      esimd::imul_impl<T, T0, T1, 1>(RmdV, Src0V, Src1V);
  rmd = RmdV[0];
  return Res[0];
}

/// Performs component-wise truncate-to-minus-infinity fraction operation of
/// \p src0. (vector version)
/// @tparam T element type of the input vector \p src0 and returned vector.
/// @tparam SZ size of the second input vector and returned vectors.
/// @param src0 the input vector.
/// @return vector of elements after fraction operation.
template <typename T, int SZ>
__ESIMD_API __ESIMD_NS::simd<T, SZ> frc(__ESIMD_NS::simd<T, SZ> src0) {
  __ESIMD_NS::simd<float, SZ> Src0 = src0;
  return __esimd_frc(Src0.data());
}

/// Performs truncate-to-minus-infinity fraction operation of \p src0.
/// (scalar version)
/// @tparam T element type of the input \p src0 and returned value.
/// @param src0 the input scalar value.
/// @return result of a fraction operation.
template <typename T> __ESIMD_API T frc(T src0) {
  __ESIMD_NS::simd<T, 1> Src0 = src0;
  __ESIMD_NS::simd<T, 1> Result = esimd::frc<T>(Src0);
  return Result[0];
}

// lzd - leading zero detection
template <typename RT, typename T0, int SZ,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<RT, SZ> lzd(__ESIMD_NS::simd<T0, SZ> src0,
                                         Sat sat = {}) {
  // Saturation parameter ignored
  __ESIMD_NS::simd<__ESIMD_NS::uint, SZ> Src0 = src0;
  return __esimd_lzd<__ESIMD_NS::uint, SZ>(Src0.data());
}

template <typename RT, typename T0, class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<RT>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T0>::value,
                             std::remove_const_t<RT>>
lzd(T0 src0, Sat sat = {}) {
  __ESIMD_NS::simd<T0, 1> Src0 = src0;
  __ESIMD_NS::simd<RT, 1> Result = esimd::lzd<RT>(Src0);
  return Result[0];
}

/// @} sycl_esimd_math

/// @addtogroup sycl_esimd_bitmanip
/// @{

/// bf_reverse
template <typename T0, typename T1, int SZ>
__ESIMD_API __ESIMD_NS::simd<T0, SZ> bf_reverse(__ESIMD_NS::simd<T1, SZ> src0) {
  __ESIMD_NS::simd<unsigned, SZ> Src0 = src0;
  return __esimd_bfrev<unsigned>(Src0.data());
}

/// bf_reverse
template <typename T0, typename T1>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value,
                             std::remove_const_t<T0>>
bf_reverse(T1 src0) {
  __ESIMD_NS::simd<T1, 1> Src0 = src0;
  __ESIMD_NS::simd<T0, 1> Result = esimd::bf_reverse<T0>(Src0);
  return Result[0];
}

/// bf_insert
template <typename T0, typename T1, int SZ, typename U, typename V, typename W>
__ESIMD_API
    std::enable_if_t<std::is_integral<T1>::value, __ESIMD_NS::simd<T0, SZ>>
    bf_insert(U src0, V src1, W src2, __ESIMD_NS::simd<T1, SZ> src3) {
  typedef typename __ESIMD_DNS::dword_type<T1> DT1;
  static_assert(std::is_integral<DT1>::value && sizeof(DT1) == sizeof(int),
                "operand conversion failed");
  __ESIMD_NS::simd<DT1, SZ> Src0 = src0;
  __ESIMD_NS::simd<DT1, SZ> Src1 = src1;
  __ESIMD_NS::simd<DT1, SZ> Src2 = src2;
  __ESIMD_NS::simd<DT1, SZ> Src3 = src3;

  return __esimd_bfi<DT1>(Src0.data(), Src1.data(), Src2.data(), Src3.data());
}

/// bf_insert
template <typename T0, typename T1, typename T2, typename T3, typename T4>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T4>::value,
                             std::remove_const_t<T0>>
bf_insert(T1 src0, T2 src1, T3 src2, T4 src3) {
  __ESIMD_NS::simd<T4, 1> Src3 = src3;
  __ESIMD_NS::simd<T0, 1> Result = esimd::bf_insert<T0>(src0, src1, src2, Src3);
  return Result[0];
}

/// bf_extract
template <typename T0, typename T1, int SZ, typename U, typename V>
__ESIMD_API
    std::enable_if_t<std::is_integral<T1>::value, __ESIMD_NS::simd<T0, SZ>>
    bf_extract(U src0, V src1, __ESIMD_NS::simd<T1, SZ> src2) {
  typedef typename __ESIMD_DNS::dword_type<T1> DT1;
  static_assert(std::is_integral<DT1>::value && sizeof(DT1) == sizeof(int),
                "operand conversion failed");
  __ESIMD_NS::simd<DT1, SZ> Src0 = src0;
  __ESIMD_NS::simd<DT1, SZ> Src1 = src1;
  __ESIMD_NS::simd<DT1, SZ> Src2 = src2;

  return __esimd_sbfe<DT1>(Src0.data(), Src1.data(), Src2.data());
}

/// bf_extract
template <typename T0, typename T1, typename T2, typename T3>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T3>::value,
                             std::remove_const_t<T0>>
bf_extract(T1 src0, T2 src1, T3 src2) {
  __ESIMD_NS::simd<T3, 1> Src2 = src2;
  __ESIMD_NS::simd<T0, 1> Result = esimd::bf_extract<T0>(src0, src1, Src2);
  return Result[0];
}

/// @} sycl_esimd_bitmanip

/// @addtogroup sycl_esimd_math
/// @{

// sincos
template <int SZ, typename U, class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<float, SZ>
sincos(__ESIMD_NS::simd<float, SZ> &dstcos, U src0, Sat sat = {}) {
  dstcos = __ESIMD_NS::cos(src0, sat);
  return __ESIMD_NS::sin(src0, sat);
}

// atan

/// @cond ESIMD_DETAIL
namespace detail {
constexpr double __ESIMD_CONST_PI = 3.1415926535897932384626433832795;
} // namespace detail
/// @endcond ESIMD_DETAIL

template <typename T, int SZ>
__ESIMD_API __ESIMD_NS::simd<T, SZ> atan(__ESIMD_NS::simd<T, SZ> src0) {
  static_assert(std::is_floating_point<T>::value,
                "Floating point argument type is expected.");
  __ESIMD_NS::simd<T, SZ> Src0 = __ESIMD_NS::abs(src0);

  __ESIMD_NS::simd<T, SZ> OneP((T)1.0);
  __ESIMD_NS::simd<T, SZ> OneN((T)-1.0);
  __ESIMD_NS::simd<T, SZ> sign;
  __ESIMD_NS::simd_mask<SZ> Gt1 = Src0 > T(1.0);

  sign.merge(OneN, OneP, src0 < 0);

  Src0.merge(__ESIMD_NS::inv(Src0), Gt1);

  __ESIMD_NS::simd<T, SZ> Src0P2 = Src0 * Src0;
  __ESIMD_NS::simd<T, SZ> Src0P4 = Src0P2 * Src0P2;

  __ESIMD_NS::simd<T, SZ> Result =
      (Src0P4 * T(0.185696) + ((Src0 * T(0.787997) + T(0.63693)) * Src0P2) +
       Src0) /
      (((((Src0 * -T(0.000121387) + T(0.00202308)) * Src0P2) +
         (Src0 * -T(0.0149145)) + T(0.182569)) *
        Src0P4) +
       ((Src0 * T(0.395889) + T(1.12158)) * Src0P2) + (Src0 * T(0.636918)) +
       T(1.0));

  Result.merge(Result - T(detail::__ESIMD_CONST_PI) / T(2.0), Gt1);

  return __ESIMD_NS::abs(Result) * sign;
}

template <typename T> __ESIMD_API T atan(T src0) {
  static_assert(std::is_floating_point<T>::value,
                "Floating point argument type is expected.");
  __ESIMD_NS::simd<T, 1> Src0 = src0;
  __ESIMD_NS::simd<T, 1> Result = esimd::atan(Src0);
  return Result[0];
}

// acos

template <typename T, int SZ>
__ESIMD_API
    std::enable_if_t<std::is_floating_point<T>::value, __ESIMD_NS::simd<T, SZ>>
    acos(__ESIMD_NS::simd<T, SZ> src0) {
  __ESIMD_NS::simd<T, SZ> Src0 = __ESIMD_NS::abs(src0);

  __ESIMD_NS::simd_mask<SZ> Neg = src0 < T(0.0);
  __ESIMD_NS::simd_mask<SZ> TooBig = Src0 >= T(0.999998);

  // Replace oversized values to ensure no possibility of sqrt of
  // a negative value later
  Src0.merge(T(0.0), TooBig);

  __ESIMD_NS::simd<T, SZ> Src01m = T(1.0) - Src0;

  __ESIMD_NS::simd<T, SZ> Src0P2 = Src01m * Src01m;
  __ESIMD_NS::simd<T, SZ> Src0P4 = Src0P2 * Src0P2;

  __ESIMD_NS::simd<T, SZ> Result =
      (((Src01m * T(0.015098965761299077) - T(0.005516443930088506)) * Src0P4) +
       ((Src01m * T(0.047654245891495528) + T(0.163910606547823220)) * Src0P2) +
       Src01m * T(2.000291665285952400) - T(0.000007239283986332)) *
      __ESIMD_NS::rsqrt(Src01m * T(2.0));

  Result.merge(T(0.0), TooBig);
  Result.merge(T(detail::__ESIMD_CONST_PI) - Result, Neg);
  return Result;
}

template <typename T>
__ESIMD_API std::enable_if_t<std::is_floating_point<T>::value, T> acos(T src0) {
  __ESIMD_NS::simd<T, 1> Src0 = src0;
  __ESIMD_NS::simd<T, 1> Result = esimd::acos(Src0);
  return Result[0];
}

// asin

template <typename T, int SZ>
__ESIMD_API
    std::enable_if_t<std::is_floating_point<T>::value, __ESIMD_NS::simd<T, SZ>>
    asin(__ESIMD_NS::simd<T, SZ> src0) {
  __ESIMD_NS::simd_mask<SZ> Neg = src0 < T(0.0);

  __ESIMD_NS::simd<T, SZ> Result =
      T(detail::__ESIMD_CONST_PI / 2.0) - esimd::acos(__ESIMD_NS::abs(src0));

  Result.merge(-Result, Neg);
  return Result;
}

template <typename T>
__ESIMD_API std::enable_if_t<std::is_floating_point<T>::value, T> asin(T src0) {
  __ESIMD_NS::simd<T, 1> Src0 = src0;
  __ESIMD_NS::simd<T, 1> Result = esimd::asin(Src0);
  return Result[0];
}
/// @} sycl_esimd_math

/// @addtogroup sycl_esimd_math
/// @{

/* atan2_fast - a fast atan2 implementation */
/* vector input */
template <int N>
__ESIMD_NS::simd<float, N> atan2_fast(__ESIMD_NS::simd<float, N> y,
                                      __ESIMD_NS::simd<float, N> x);
/* scalar input */
template <typename T> float atan2_fast(T y, T x);

/* atan2 - atan2 implementation */
/* For Vector input */
template <int N>
__ESIMD_NS::simd<float, N> atan2(__ESIMD_NS::simd<float, N> y,
                                 __ESIMD_NS::simd<float, N> x);
/* scalar Input */
template <typename T> float atan2(T y, T x);

/* fmod: */
/* vector input */
template <int N>
__ESIMD_NS::simd<float, N> fmod(__ESIMD_NS::simd<float, N> y,
                                __ESIMD_NS::simd<float, N> x);
/* scalar Input */
template <typename T> float fmod(T y, T x);

/* sin_emu - EU emulation for sin(x) */
/* For Vector input */
template <int N>
__ESIMD_NS::simd<float, N> sin_emu(__ESIMD_NS::simd<float, N> x);
/* scalar Input */
template <typename T> float sin_emu(T x);

/* cos_emu - EU emulation for cos(x) */
/* For Vector input */
template <int N>
__ESIMD_NS::simd<float, N> cos_emu(__ESIMD_NS::simd<float, N> x);

/* scalar Input */
template <typename T> float cos_emu(T x);

/* tanh_cody_waite - Cody-Waite implementation for tanh(x) */
/* float input */
float tanh_cody_waite(float x);
/* vector input */
template <int N>
__ESIMD_NS::simd<float, N> tanh_cody_waite(__ESIMD_NS::simd<float, N> x);
/* tanh - opencl like implementation for tanh(x) */
/* float input */
float tanh(float x);
/* vector input */
template <int N> __ESIMD_NS::simd<float, N> tanh(__ESIMD_NS::simd<float, N> x);

/* ------------------------- Extended Math Routines
 * -------------------------------------------------*/

// For vector input
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N>
atan2_fast(__ESIMD_NS::simd<float, N> y, __ESIMD_NS::simd<float, N> x) {
  /* smallest such that 1.0+CONST_DBL_EPSILON != 1.0 */
  constexpr float CONST_DBL_EPSILON = 0.00001f;
  __ESIMD_NS::simd<float, N> OneP(1.0f);
  __ESIMD_NS::simd<float, N> OneN(-1.0f);
  __ESIMD_NS::simd<float, N> sign;
  __ESIMD_NS::simd<float, N> atan2;
  __ESIMD_NS::simd<float, N> r;
  __ESIMD_NS::simd_mask<N> mask = x < 0;
  __ESIMD_NS::simd<float, N> abs_y = __ESIMD_NS::abs(y) + CONST_DBL_EPSILON;

  r.merge((x + abs_y) / (abs_y - x), (x - abs_y) / (x + abs_y), mask);
  atan2.merge(float(detail::__ESIMD_CONST_PI) * 0.75f,
              float(detail::__ESIMD_CONST_PI) * 0.25f, mask);
  atan2 += (0.1963f * r * r - 0.9817f) * r;

  sign.merge(OneN, OneP, y < 0);

  return atan2 * sign;
}

//   For Scalar Input
template <> ESIMD_INLINE float atan2_fast(float y, float x) {
  __ESIMD_NS::simd<float, 1> vy = y;
  __ESIMD_NS::simd<float, 1> vx = x;
  __ESIMD_NS::simd<float, 1> atan2 = esimd::atan2_fast(vy, vx);
  return atan2[0];
}

// atan2
// For Vector input
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N> atan2(__ESIMD_NS::simd<float, N> y,
                                              __ESIMD_NS::simd<float, N> x) {
  __ESIMD_NS::simd<float, N> atan2;
  __ESIMD_NS::simd_mask<N> mask;
  __ESIMD_NS::simd<float, N> atan = esimd::atan(y / x);

  constexpr float CONST_DBL_EPSILON = 0.00001f;

  mask = (__ESIMD_NS::abs(x) < CONST_DBL_EPSILON && y < -CONST_DBL_EPSILON);
  atan2.merge(float(-detail::__ESIMD_CONST_PI) / 2.f, 0.f, mask);
  mask = (__ESIMD_NS::abs(x) < CONST_DBL_EPSILON && y > CONST_DBL_EPSILON);
  atan2.merge(float(detail::__ESIMD_CONST_PI) / 2.f, mask);
  mask = (x < -CONST_DBL_EPSILON && y < -CONST_DBL_EPSILON);
  atan2.merge(atan - float(detail::__ESIMD_CONST_PI), mask);
  mask = (x < -CONST_DBL_EPSILON && y >= -CONST_DBL_EPSILON);
  atan2.merge(atan + float(detail::__ESIMD_CONST_PI), mask);
  mask = (x > CONST_DBL_EPSILON);
  atan2.merge(atan, mask);

  return atan2;
}

// For Scalar Input
template <> ESIMD_INLINE float atan2(float y, float x) {
  __ESIMD_NS::simd<float, 1> vy = y;
  __ESIMD_NS::simd<float, 1> vx = x;
  __ESIMD_NS::simd<float, 1> atan2 = esimd::atan2(vy, vx);
  return atan2[0];
}

// fmod:
// For Vector input
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N> fmod(__ESIMD_NS::simd<float, N> y,
                                             __ESIMD_NS::simd<float, N> x) {
  __ESIMD_NS::simd<float, N> abs_x = __ESIMD_NS::abs(x);
  __ESIMD_NS::simd<float, N> abs_y = __ESIMD_NS::abs(y);

  auto fmod_sign_mask = (y.template bit_cast_view<int32_t>()) & 0x80000000;

  __ESIMD_NS::simd<float, N> reminder =
      abs_y - abs_x * __ESIMD_NS::trunc<float>(abs_y / abs_x);

  abs_x.merge(0.0f, reminder >= 0);
  __ESIMD_NS::simd<float, N> fmod = reminder + abs_x;
  __ESIMD_NS::simd<float, N> fmod_abs = __ESIMD_NS::abs(fmod);

  auto fmod_bits =
      (fmod_abs.template bit_cast_view<int32_t>()) | fmod_sign_mask;
  return fmod_bits.template bit_cast_view<float>();
}

// For Scalar Input
template <> ESIMD_INLINE float fmod(float y, float x) {
  return fmod(__ESIMD_NS::simd<float, 1>(y), __ESIMD_NS::simd<float, 1>(x))[0];
}

// sin_emu - EU emulation for sin(x)
// For Vector input
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N> sin_emu(__ESIMD_NS::simd<float, N> x) {
  __ESIMD_NS::simd<float, N> x1;
  __ESIMD_NS::simd<float, N> x2;
  __ESIMD_NS::simd<float, N> t3;

  __ESIMD_NS::simd<float, N> sign;
  __ESIMD_NS::simd<float, N> fTrig;
  __ESIMD_NS::simd<float, N> TwoPI(float(detail::__ESIMD_CONST_PI) * 2.0f);
  __ESIMD_NS::simd<float, N> CmpI((float)detail::__ESIMD_CONST_PI);
  __ESIMD_NS::simd<float, N> OneP(1.0f);
  __ESIMD_NS::simd<float, N> OneN(-1.0f);

  x = esimd::fmod(x, TwoPI);
  x.merge(TwoPI + x, x < 0);

  x1.merge(CmpI - x, x - CmpI, (x <= float(detail::__ESIMD_CONST_PI)));
  x1.merge(x, (x <= float(detail::__ESIMD_CONST_PI) * 0.5f));
  x1.merge(TwoPI - x, (x > float(detail::__ESIMD_CONST_PI) * 1.5f));

  sign.merge(OneN, OneP, (x > float(detail::__ESIMD_CONST_PI)));

  x2 = x1 * x1;
  t3 = x2 * x1 * 0.1666667f;

  fTrig =
      x1 + t3 * (OneN + x2 * 0.05f *
                            (OneP + x2 * 0.0238095f *
                                        (OneN + x2 * 0.0138889f *
                                                    (OneP - x2 * 0.0090909f))));
  fTrig *= sign;
  return fTrig;
}

// scalar Input
template <> ESIMD_INLINE float sin_emu(float x0) {
  return esimd::sin_emu(__ESIMD_NS::simd<float, 1>(x0))[0];
}

// cos_emu - EU emulation for sin(x)
// For Vector input
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N> cos_emu(__ESIMD_NS::simd<float, N> x) {
  return esimd::sin_emu(0.5f * float(detail::__ESIMD_CONST_PI) - x);
}

// scalar Input
template <> ESIMD_INLINE float cos_emu(float x0) {
  return esimd::cos_emu(__ESIMD_NS::simd<float, 1>(x0))[0];
}

/// @cond ESIMD_DETAIL
namespace detail {

template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N>
tanh_cody_waite_impl(__ESIMD_NS::simd<float, N> x) {
  /*
   *      0           x_small             x_medium            x_large
   *  |   x   | rational polynomial | 1 - 2/(1 + exp(2*x)) |  1
   *
   * rational polynomial for single precision = x + x * (g * (p[1] * g + p[0]) /
   * (g + q[0]) g = x^2 p0 = -0.82377 28127 E+00 p1 = -0.38310 10665 E-02 q0 =
   * 0.24713 19654 E+01 q1 = 1.00000 00000 E+00
   *
   */

  constexpr float p0 = -0.8237728127E+00f;
  constexpr float p1 = -0.3831010665E-02f;
  constexpr float q0 = 0.2471319654E+01f;
  constexpr float q1 = 1.0000000000E+00f;
  constexpr float xsmall = 4.22863966691620432990E-04f;
  constexpr float xmedium = 0.54930614433405484570f;
  constexpr float xlarge = 8.66433975699931636772f;

  using RT = __ESIMD_NS::simd<float, N>;

  RT absX = __ESIMD_NS::abs(x);
  RT g = absX * absX;

  RT sign;
  sign.merge(-1.f, 1.f, x < 0.f);

  auto isLarge = absX > xlarge;
  auto minor = absX <= xlarge;
  auto isGtMed = minor & (absX > xmedium);
  auto isGtSmall = (absX > xsmall) & (absX <= xmedium);

  RT res;
  res.merge(sign, x, isLarge);
  auto temp = __ESIMD_NS::exp(absX * 2.0f) + 1.f;
  temp = ((temp - 2.f) / temp) * sign;
  res.merge(temp, isGtMed);
  res.merge((absX + absX * g * (g * p1 + p0) / (g + q0)) * sign, isGtSmall);

  return res;
}

template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N>
tanh_impl(__ESIMD_NS::simd<float, N> x) {
  /*
   *      0                       x_small                          x_large
   * |    x    |  ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )  |      1
   *
   */

  constexpr float xsmall = 0.000045f; // same as exp(-10.0f)
  constexpr float xlarge = 40.f;

  using RT = __ESIMD_NS::simd<float, N>;

  RT absX = __ESIMD_NS::abs(x);

  RT sign;
  sign.merge(-1.f, 1.f, x < 0.f);

  auto isLarge = (absX > xlarge);
  auto isLessE = (absX <= xlarge);

  RT res;
  res.merge(sign, x, isLarge);

  RT exp;
  exp = __ESIMD_NS::exp(absX * 2.f);

  res.merge(((exp - 1.f) / (exp + 1.f)) * sign, (absX > xsmall) & isLessE);

  return res;
}
} // namespace detail
/// @endcond ESIMD_DETAIL

/* tanh_cody_waite - Cody-Waite implementation for tanh(x) */
/* float input */
ESIMD_INLINE float tanh_cody_waite(float x) {
  return detail::tanh_cody_waite_impl(__ESIMD_NS::simd<float, 1>(x))[0];
}
/* vector input */
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N>
tanh_cody_waite(__ESIMD_NS::simd<float, N> x) {
  return detail::tanh_cody_waite_impl(x);
}

/* tanh - opencl like implementation for tanh(x) */
/* float input */
ESIMD_INLINE float tanh(float x) {
  return esimd::detail::tanh_impl(__ESIMD_NS::simd<float, 1>(x))[0];
}
/* vector input */
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N> tanh(__ESIMD_NS::simd<float, N> x) {
  return esimd::detail::tanh_impl(x);
}

template <typename T, int N>
__ESIMD_NS::simd<T, N> dp4(__ESIMD_NS::simd<T, N> v1,
                           __ESIMD_NS::simd<T, N> v2) {
  auto retv = __esimd_dp4<T, N>(v1.data(), v2.data());
  return retv;
}

/// srnd - perform stochastic rounding.
/// Supported conversions:
///   float -> half
/// Available on PVC_XT+
/// \param src0 the operand to be rounded
/// \param src1 random number used for rounding
/// \return the converted value
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<sycl::half, N>
srnd(__ESIMD_NS::simd<float, N> src0, __ESIMD_NS::simd<uint16_t, N> src1) {
  return __esimd_srnd<N>(src0.data(), src1.data());
}

/// @} sycl_esimd_math

/// @addtogroup sycl_esimd_logical
/// @{

/// Performs a fused multiply add computation with three vector operands.
/// @tparam T type of the vector operands.
/// @tparam N size of the vector operands.
/// @param a First vector function argument.
/// @param b Second vector function argument.
/// @param c Third vector function argument.
/// @return the computation result
template <typename T, int N>
ESIMD_INLINE __ESIMD_NS::simd<T, N> fma(__ESIMD_NS::simd<T, N> a,
                                        __ESIMD_NS::simd<T, N> b,
                                        __ESIMD_NS::simd<T, N> c) {
  static_assert(__ESIMD_DNS::is_generic_floating_point_v<T>,
                "fma only supports floating point types");
  using CppT = __ESIMD_DNS::element_type_traits<T>::EnclosingCppT;
  auto Ret = __spirv_ocl_fma<__ESIMD_DNS::__raw_t<CppT>, N>(
      __ESIMD_DNS::convert_vector<CppT, T, N>(a.data()),
      __ESIMD_DNS::convert_vector<CppT, T, N>(b.data()),
      __ESIMD_DNS::convert_vector<CppT, T, N>(c.data()));
  return __ESIMD_DNS::convert_vector<T, CppT, N>(Ret);
}

/// @} sycl_esimd_logical

} // namespace ext::intel::experimental::esimd
} // namespace _V1
} // namespace sycl
