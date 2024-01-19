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

/// Shift left operation (vector version)
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vector.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted left values.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             __ESIMD_NS::simd<T0, SZ>>
shl(__ESIMD_NS::simd<T1, SZ> src0, U src1, Sat sat = {}) {
  using ComputationTy =
      __ESIMD_DNS::computation_type_t<decltype(src0), int32_t>;
  ComputationTy Src0 = src0;
  ComputationTy Src1 = src1;

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_on_tag>) {
    if constexpr (std::is_unsigned<T0>::value) {
      if constexpr (std::is_unsigned<
                        typename ComputationTy::element_type>::value)
        return __esimd_uushl_sat<T0, typename ComputationTy::element_type, SZ>(
            Src0.data(), Src1.data());
      else
        return __esimd_usshl_sat<T0, typename ComputationTy::element_type, SZ>(
            Src0.data(), Src1.data());
    } else {
      if constexpr (std::is_signed<typename ComputationTy::element_type>::value)
        return __esimd_sushl_sat<T0, typename ComputationTy::element_type, SZ>(
            Src0.data(), Src1.data());
      else
        return __esimd_ssshl_sat<T0, typename ComputationTy::element_type, SZ>(
            Src0.data(), Src1.data());
    }
  } else {
    if constexpr (std::is_unsigned<T0>::value) {
      if constexpr (std::is_unsigned<
                        typename ComputationTy::element_type>::value)
        return __esimd_uushl<T0, typename ComputationTy::element_type, SZ>(
            Src0.data(), Src1.data());
      else
        return __esimd_usshl<T0, typename ComputationTy::element_type, SZ>(
            Src0.data(), Src1.data());
    } else {
      if constexpr (std::is_signed<typename ComputationTy::element_type>::value)
        return __esimd_sushl<T0, typename ComputationTy::element_type, SZ>(
            Src0.data(), Src1.data());
      else
        return __esimd_ssshl<T0, typename ComputationTy::element_type, SZ>(
            Src0.data(), Src1.data());
    }
  }
}

/// Shift left operation (scalar version)
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted left value.
template <typename T0, typename T1, typename T2,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T2>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<T2>::value,
                             std::remove_const_t<T0>>
shl(T1 src0, T2 src1, Sat sat = {}) {
  __ESIMD_NS::simd<T1, 1> Src0 = src0;
  __ESIMD_NS::simd<T0, 1> Result =
      esimd::shl<T0, T1, 1, T2, Sat>(Src0, src1, sat);
  return Result[0];
}

/// Logical Shift Right (vector version)
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted elements.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             __ESIMD_NS::simd<T0, SZ>>
lsr(__ESIMD_NS::simd<T1, SZ> src0, U src1, Sat sat = {}) {
  using IntermedTy = __ESIMD_DNS::computation_type_t<T1, T1>;
  typedef typename std::make_unsigned<IntermedTy>::type ComputationTy;
  __ESIMD_NS::simd<ComputationTy, SZ> Src0 = src0;
  __ESIMD_NS::simd<ComputationTy, SZ> Src1 = src1;
  // TODO H/W supports saturation with this op - map to more efficient version.
  __ESIMD_NS::simd<ComputationTy, SZ> Result = Src0.data() >> Src1.data();

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T0>(Result);
}

/// Logical Shift Right (scalar version)
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value \p src0. Must be any integer
/// type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted value.
template <typename T0, typename T1, typename T2,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T2>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<T2>::value,
                             std::remove_const_t<T0>>
lsr(T1 src0, T2 src1, Sat sat = {}) {
  __ESIMD_NS::simd<T1, 1> Src0 = src0;
  __ESIMD_NS::simd<T0, 1> Result =
      esimd::lsr<T0, T1, 1, T2, Sat>(Src0, src1, sat);

  return Result[0];
}

/// Arithmetical Shift Right (vector version)
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted elements.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             __ESIMD_NS::simd<T0, SZ>>
asr(__ESIMD_NS::simd<T1, SZ> src0, U src1, Sat sat = {}) {
  using IntermedTy = __ESIMD_DNS::computation_type_t<T1, T1>;
  typedef typename std::make_signed<IntermedTy>::type ComputationTy;
  __ESIMD_NS::simd<ComputationTy, SZ> Src0 = src0;
  // TODO H/W supports saturation with this op - map to more efficient version.
  __ESIMD_NS::simd<ComputationTy, SZ> Result = Src0 >> src1;

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T0>(Result);
}

/// Arithmetical Shift Right (scalar version)
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value \p src0. Must be any integer
/// type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted value.
template <typename T0, typename T1, typename T2,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T2>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<T2>::value,
                             std::remove_const_t<T0>>
asr(T1 src0, T2 src1, Sat sat = {}) {
  __ESIMD_NS::simd<T1, 1> Src0 = src0;
  __ESIMD_NS::simd<T0, 1> Result =
      esimd::asr<T0, T1, 1, T2, Sat>(Src0, src1, sat);
  return Result[0];
}

/// Shift right operation (vector version)
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vector.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted right values.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             __ESIMD_NS::simd<T0, SZ>>
shr(__ESIMD_NS::simd<T1, SZ> src0, U src1, Sat sat = {}) {
  if constexpr (std::is_unsigned<T1>::value) {
    return esimd::lsr<T0, T1, SZ, U, Sat>(src0, src1, sat);
  } else {
    return esimd::asr<T0, T1, SZ, U, Sat>(src0, src1, sat);
  }
}

/// Shift right operation (scalar version)
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted right value.
template <typename T0, typename T1, typename T2,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T2>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<T2>::value,
                             std::remove_const_t<T0>>
shr(T1 src0, T2 src1, Sat sat = {}) {
  __ESIMD_NS::simd<T1, 1> Src0 = src0;
  __ESIMD_NS::simd<T0, 1> Result =
      esimd::shr<T0, T1, 1, T2, Sat>(Src0, src1, sat);
  return Result[0];
}

/// Rotate left operation with two vector inputs
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the vector with number of bit positions by which the elements of
/// the input vector \p src0 shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ>
__ESIMD_API std::enable_if_t<
    __ESIMD_NS::detail::is_type<T0, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<T1, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>(),
    __ESIMD_NS::simd<T0, SZ>>
rol(__ESIMD_NS::simd<T1, SZ> src0, __ESIMD_NS::simd<T1, SZ> src1) {
  return __esimd_rol<T0, T1, SZ>(src0.data(), src1.data());
}

/// Rotate left operation with a vector and a scalar inputs
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API std::enable_if_t<
    __ESIMD_NS::detail::is_type<T0, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<T1, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<U, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>(),
    __ESIMD_NS::simd<T0, SZ>>
rol(__ESIMD_NS::simd<T1, SZ> src0, U src1) {
  __ESIMD_NS::simd<T1, SZ> Src1 = src1;
  return esimd::rol<T0>(src0, Src1);
}

/// Rotate left operation with two scalar inputs
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return rotated left value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_esimd_scalar<T0>::value &&
        __ESIMD_DNS::is_esimd_scalar<T1>::value &&
        __ESIMD_DNS::is_esimd_scalar<T2>::value &&
        __ESIMD_NS::detail::is_type<T0, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<T1, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<T2, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>(),
    std::remove_const_t<T0>>
rol(T1 src0, T2 src1) {
  __ESIMD_NS::simd<T1, 1> Src0 = src0;
  __ESIMD_NS::simd<T0, 1> Result = esimd::rol<T0, T1, 1, T2>(Src0, src1);
  return Result[0];
}

/// Rotate right operation with two vector inputs
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the vector with number of bit positions by which the elements of
/// the input vector \p src0 shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ>
__ESIMD_API std::enable_if_t<
    __ESIMD_NS::detail::is_type<T0, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<T1, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>(),
    __ESIMD_NS::simd<T0, SZ>>
ror(__ESIMD_NS::simd<T1, SZ> src0, __ESIMD_NS::simd<T1, SZ> src1) {
  return __esimd_ror<T0, T1, SZ>(src0.data(), src1.data());
}

/// Rotate right operation with a vector and a scalar inputs
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API std::enable_if_t<
    __ESIMD_NS::detail::is_type<T0, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<T1, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<U, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>(),
    __ESIMD_NS::simd<T0, SZ>>
ror(__ESIMD_NS::simd<T1, SZ> src0, U src1) {
  __ESIMD_NS::simd<T1, SZ> Src1 = src1;
  return esimd::ror<T0>(src0, Src1);
}

/// Rotate right operation with two scalar inputs
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return rotated right value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_esimd_scalar<T0>::value &&
        __ESIMD_DNS::is_esimd_scalar<T1>::value &&
        __ESIMD_DNS::is_esimd_scalar<T2>::value &&
        __ESIMD_NS::detail::is_type<T0, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<T1, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>() &&
        __ESIMD_NS::detail::is_type<T2, int16_t, uint16_t, int32_t, uint32_t,
                                    int64_t, uint64_t>(),
    std::remove_const_t<T0>>
ror(T1 src0, T2 src1) {
  __ESIMD_NS::simd<T1, 1> Src0 = src0;
  __ESIMD_NS::simd<T0, 1> Result = esimd::ror<T0, T1, 1, T2>(Src0, src1);
  return Result[0];
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

template <int N>
__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::addc(carry, src0, src1);")
__ESIMD_API __ESIMD_NS::simd<uint32_t, N> addc(
    __ESIMD_NS::simd<uint32_t, N> &carry, __ESIMD_NS::simd<uint32_t, N> src0,
    __ESIMD_NS::simd<uint32_t, N> src1) {
  return __ESIMD_NS::addc(carry, src0, src1);
}

template <int N>
__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::addc(carry, src0, src1);")
__ESIMD_API __ESIMD_NS::simd<uint32_t, N> addc(
    __ESIMD_NS::simd<uint32_t, N> &carry, __ESIMD_NS::simd<uint32_t, N> src0,
    uint32_t src1) {
  return __ESIMD_NS::addc(carry, src0, src1);
}

template <int N>
__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::addc(carry, src0, src1);")
__ESIMD_API __ESIMD_NS::simd<uint32_t, N> addc(
    __ESIMD_NS::simd<uint32_t, N> &carry, uint32_t src0,
    __ESIMD_NS::simd<uint32_t, N> src1) {
  return __ESIMD_NS::addc(carry, src0, src1);
}

__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::addc(carry, src0, src1);")
__ESIMD_API uint32_t addc(uint32_t &carry, uint32_t src0, uint32_t src1) {
  return __ESIMD_NS::addc(carry, src0, src1);
}

template <int N>
__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::subb(borrow, src0, src1);")
__ESIMD_API __ESIMD_NS::simd<uint32_t, N> subb(
    __ESIMD_NS::simd<uint32_t, N> &borrow, __ESIMD_NS::simd<uint32_t, N> src0,
    __ESIMD_NS::simd<uint32_t, N> src1) {
  return __ESIMD_NS::subb(borrow, src0, src1);
}

template <int N>
__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::subb(borrow, src0, src1);")
__ESIMD_API __ESIMD_NS::simd<uint32_t, N> subb(
    __ESIMD_NS::simd<uint32_t, N> &borrow, __ESIMD_NS::simd<uint32_t, N> src0,
    uint32_t src1) {
  return __ESIMD_NS::subb(borrow, src0, src1);
}

template <int N>
__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::subb(borrow, src0, src1);")
__ESIMD_API __ESIMD_NS::simd<uint32_t, N> subb(
    __ESIMD_NS::simd<uint32_t, N> &borrow, uint32_t src0,
    __ESIMD_NS::simd<uint32_t, N> src1) {
  return __ESIMD_NS::subb(borrow, src0, src1);
}

__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::subb(borrow, src0, src1);")
__ESIMD_API uint32_t subb(uint32_t &borrow, uint32_t src0, uint32_t src1) {
  return __ESIMD_NS::subb(borrow, src0, src1);
}

/// Integral quotient (vector version)
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1.
/// @param src0 the dividend input vector.
/// @param src1 the divisor scalar value.
/// @return vector of quotient elements.
template <typename T, int SZ, typename U>
__SYCL_DEPRECATED("Use: src0 / src1;")
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                     __ESIMD_NS::simd<T, SZ>> quot(__ESIMD_NS::simd<T, SZ> src0,
                                                   U src1) {
  return src0 / src1;
}

/// Integral quotient (scalar version)
/// @tparam T0 element type of the dividend \p src0 and returned value.
/// @tparam T1 element type of the divisor \p src1.
/// @param src0 the dividend.
/// @param src1 the divisor.
/// @return quotient value.
template <typename T0, typename T1>
__SYCL_DEPRECATED("Use: src0 / src1;")
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value,
                             std::remove_const_t<T0>> quot(T0 src0, T1 src1) {
  return src0 / src1;
}

/// Modulo (vector version)
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1.
/// @param src0 the dividend input vector.
/// @param src1 the divisor scalar value.
/// @return vector of elements after applying modulo operation.
template <typename T, int SZ, typename U>
__SYCL_DEPRECATED("Use: src0 % src1;")
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                     __ESIMD_NS::simd<T, SZ>> mod(__ESIMD_NS::simd<T, SZ> src0,
                                                  U src1) {
  return src0 % src1;
}

/// Modulo (scalar version)
/// @tparam T0 element type of the dividend \p src0 and returned value.
/// @tparam T1 element type of the divisor \p src1.
/// @param src0 the dividend.
/// @param src1 the divisor.
/// @return Modulo value.
template <typename T0, typename T1>
__SYCL_DEPRECATED("Use: src0 % src1;")
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value,
                             std::remove_const_t<T0>> mod(T0 src0, T1 src1) {
  return src0 % src1;
}

/// Integral division with a vector dividend and a scalar divisor. Computes
/// quotient and remainder of division.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1.
/// @param[out] remainder the vector of remainders from a division operation.
/// @param src0 the dividend input vector.
/// @param src1 the divisor scalar value.
/// @return vector of quotient elements.
template <typename T, int SZ, typename U>
__SYCL_DEPRECATED("Use: T res = src0 / src1; T remainder = src0 % src1;")
__ESIMD_API std::enable_if_t<
    std::is_integral<T>::value && std::is_integral<U>::value,
    __ESIMD_NS::simd<T, SZ>> div(__ESIMD_NS::simd<T, SZ> &remainder,
                                 __ESIMD_NS::simd<T, SZ> src0, U src1) {
  remainder = src0 % src1;
  return src0 / src1;
}

/// Integral division with a scalar dividend and a vector divisor. Computes
/// quotient and remainder of division.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1.
/// @param[out] remainder the vector of remainders from a division operation.
/// @param src0 the dividend scalar value.
/// @param src1 the divisor input vector.
/// @return vector of quotient elements.
template <typename T, int SZ, typename U>
__SYCL_DEPRECATED("Use: T res = src0 / src1; T remainder = src0 % src1;")
__ESIMD_API std::enable_if_t<
    std::is_integral<T>::value && std::is_integral<U>::value &&
        __ESIMD_DNS::is_esimd_scalar<U>::value,
    __ESIMD_NS::simd<T, SZ>> div(__ESIMD_NS::simd<T, SZ> &remainder, U src0,
                                 __ESIMD_NS::simd<T, SZ> src1) {
  remainder = src0 % src1;
  return src0 / src1;
}

/// Integral division (scalar version). Computes quotient and remainder of
/// division.
/// @tparam RT element type of the output remainder vector.
/// @tparam T0 element type of the dividend \p src0.
/// @tparam T1 element type of the divisor \p src1.
/// @param[out] remainder the vector of size 1 with a remainder from division.
/// @param src0 the dividend scalar value.
/// @param src1 the divisor scalar value.
/// @return scalar quotient value.
template <typename RT, typename T0, typename T1>
__SYCL_DEPRECATED("Use: T res = src0 / src1; T remainder = src0 % src1;")
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_esimd_scalar<RT>::value &&
        __ESIMD_DNS::is_esimd_scalar<T0>::value &&
        __ESIMD_DNS::is_esimd_scalar<T1>::value,
    std::remove_const_t<RT>> div(__ESIMD_NS::simd<std::remove_const_t<RT>, 1>
                                     &remainder,
                                 T0 src0, T1 src1) {
  remainder[0] = src0 % src1;
  return src0 / src1;
}

// Dot product builtins
#if defined(ESIMD_GEN7_5) || defined(ESIMD_GEN8) || defined(ESIMD_GEN8_5) ||   \
    defined(ESIMD_GEN9) || defined(ESIMD_GEN9_5)

/// Dot product on groups of 4 elements.
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// @tparam U type of scalar operand \p src1.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API __ESIMD_NS::simd<T0, SZ> dp2(__ESIMD_NS::simd<T1, SZ> src0, U src1,
                                         Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");
  __ESIMD_NS::simd<float, SZ> Src0 = src0;
  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result = __esimd_dp2(Src0.data(), Src1.data());
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T0>(Result);
}

/// Dot product on groups of 4 elements.
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// @tparam U type of scalar operand \p src1.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API __ESIMD_NS::simd<T0, SZ> dp3(__ESIMD_NS::simd<T1, SZ> src0, U src1,
                                         Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");
  __ESIMD_NS::simd<float, SZ> Src0 = src0;
  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result = __esimd_dp3(Src0.data(), Src1.data());
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T0>(Result);
}

/// Dot product on groups of 4 elements.
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// @tparam U type of scalar operand \p src1.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API __ESIMD_NS::simd<T0, SZ> dp4(__ESIMD_NS::simd<T1, SZ> src0, U src1,
                                         Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");
  __ESIMD_NS::simd<float, SZ> Src0 = src0;
  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result = __esimd_dp4(Src0.data(), Src1.data());
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T0>(Result);
}

/// Dot product on groups of 4 elements.
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// @tparam U type of scalar operand \p src1.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, typename U, int SZ,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API __ESIMD_NS::simd<T0, SZ> dph(__ESIMD_NS::simd<T1, SZ> src0, U src1,
                                         Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");
  __ESIMD_NS::simd<float, SZ> Src0 = src0;
  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result = __esimd_dph(Src0.data(), Src1.data());
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T0>(Result);
}

/// Linear equation.
/// @tparam RT element type of the output vector.
/// @tparam T1 element type of the first input vector \p src0.
/// @tparam T2 element type of the second input vector \p src1.
/// @tparam SZ size of the second input vector and returned vectors. Must be a
/// multiple of 4.
/// @param src0 the first input vector of size 4.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return resulting vector from linear equation operation.
template <typename RT, typename T1, typename T2, int SZ,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API __ESIMD_NS::simd<RT, SZ> line(__ESIMD_NS::simd<T1, 4> src0,
                                          __ESIMD_NS::simd<T2, SZ> src1,
                                          Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  __ESIMD_NS::simd<float, 4> Src0 = src0;
  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result = __esimd_line(Src0.data(), Src1.data());

  __ESIMD_NS::simd<RT, SZ> Result;
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<RT>(Result);
}

/// Linear equation.
/// @tparam RT element type of the output vector.
/// @tparam T element type of the first input vector \p src0.
/// @tparam SZ size of the second input vector and returned vectors. Must be a
/// multiple of 4.
/// @param P the first input value.
/// @param Q the second input value.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return resulting vector from linear equation operation.
template <typename RT, typename T, int SZ,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API __ESIMD_NS::simd<RT, SZ> line(float P, float Q,
                                          __ESIMD_NS::simd<T, SZ> src1,
                                          Sat sat = {}) {
  __ESIMD_NS::simd<float, 4> Src0 = P;
  Src0(3) = Q;
  return esimd::line<RT>(Src0, src1, sat);
}

#else
// The old implementation is to generate vISA IRs for dp2/dp3/dp4/dph/line.
// Now We change to use direct mul/add, and hope to generate mad instructions
// at the end, to still get the performance as good as HW solution.
// We rely on "pragma unroll" to get better code.
// The only input and return types for these APIs are floats.
// In order to be able to use the old emu code, we keep the template argument
// for the type, although the type "T" can only be float.
// We use std::enable_if to force the float type only.
// If the gen is not specified we warn the programmer that they are potentially
// using a less efficient implementation if not on GEN10 or above.

/// Dot product on groups of 4 elements.
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector. Must be a float type.
/// @tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// @tparam U type of scalar operand \p src1. Must be a float type.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_fp_or_dword_type<T1>::value &&
        std::is_floating_point<T1>::value &&
        __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
        std::is_floating_point<U>::value,
    __ESIMD_NS::simd<T0, SZ>> dp2(__ESIMD_NS::simd<T1, SZ> src0, U src1,
                                  Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[i] * Src1[i] + src0[i + 1] * Src1[i + 1];
  }
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T1>(Result);
}

/// Dot product on groups of 4 elements.
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector. Must be a float type.
/// @tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// @tparam U type of scalar operand \p src1. Must be a float type.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_fp_or_dword_type<T1>::value &&
        std::is_floating_point<T1>::value &&
        __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
        std::is_floating_point<U>::value,
    __ESIMD_NS::simd<T0, SZ>> dp3(__ESIMD_NS::simd<T1, SZ> src0, U src1,
                                  Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[i] * Src1[i] + src0[i + 1] * Src1[i + 1] +
                             src0[i + 2] * Src1[i + 2];
  }
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T1>(Result);
}

/// Dot product on groups of 4 elements.
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector. Must be a float type.
/// @tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// @tparam U type of scalar operand \p src1. Must be a float type.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_fp_or_dword_type<T1>::value &&
        std::is_floating_point<T1>::value &&
        __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
        std::is_floating_point<U>::value,
    __ESIMD_NS::simd<T0, SZ>> dp4(__ESIMD_NS::simd<T1, SZ> src0, U src1,
                                  Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  __ESIMD_NS::simd<T1, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[i] * Src1[i] + src0[i + 1] * Src1[i + 1] +
                             src0[i + 2] * Src1[i + 2] +
                             src0[i + 3] * Src1[i + 3];
  }
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T1>(Result);
}

/// Dot product on groups of 4 elements.
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector. Must be a float type.
/// @tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// @tparam U type of scalar operand \p src1. Must be a float type.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T, typename U, int SZ,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T>::value &&
                         std::is_floating_point<T>::value &&
                         __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
                         std::is_floating_point<U>::value,
                     __ESIMD_NS::simd<T, SZ>> dph(__ESIMD_NS::simd<T, SZ> src0,
                                                  U src1, Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[i] * Src1[i] + src0[i + 1] * Src1[i + 1] +
                             src0[i + 2] * Src1[i + 2] + 1.0 * Src1[i + 3];
  }
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T>(Result);
}

/// Linear equation.
/// @tparam T element type of the second input vector \p src1 and returned
/// vector. Must be a float type.
/// @tparam SZ size of the second input vector and returned vectors.
/// Must be a multiple of 4.
/// @param src0 the first input vector of size 4.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return resulting vector from linear equation operation.
template <typename T, int SZ, class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T>::value &&
                         std::is_floating_point<T>::value,
                     __ESIMD_NS::simd<T, SZ>> line(__ESIMD_NS::simd<T, 4> src0,
                                                   __ESIMD_NS::simd<T, SZ> src1,
                                                   Sat sat = {}) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  __ESIMD_NS::simd<T, SZ> Src1 = src1;
  __ESIMD_NS::simd<T, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[0] * src1[i] + src0[3];
  }

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T>(Result);
}

/// Linear equation.
/// @tparam T element type of the first input vector \p src0. Must be a float
/// type.
/// @tparam SZ size of the second input vector and returned vectors. Must
/// be a multiple of 4.
/// @param P the first input value.
/// @param Q the second input value.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return resulting vector from linear equation operation.
template <typename T, int SZ, class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T>::value &&
                         std::is_floating_point<T>::value,
                     __ESIMD_NS::simd<T, SZ>> line(float P, float Q,
                                                   __ESIMD_NS::simd<T, SZ> src1,
                                                   Sat sat = {}) {
  __ESIMD_NS::simd<T, 4> Src0 = P;
  Src0(3) = Q;
  return esimd::line<T>(Src0, src1, sat);
}

#endif

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

// lrp
#if defined(ESIMD_GEN7_5) || defined(ESIMD_GEN8) || defined(ESIMD_GEN8_5) ||   \
    defined(ESIMD_GEN9) || defined(ESIMD_GEN9_5)

template <int SZ, typename U, typename V,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API __ESIMD_NS::simd<float, SZ> lrp(__ESIMD_NS::simd<float, SZ> src0,
                                            U src1, V src2, Sat sat = {}) {
  static_assert(SZ >= 4 && (SZ & 0x3) == 0,
                "vector size must be a multiple of 4");
  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Src2 = src2;
  __ESIMD_NS::simd<float, SZ> Result =
      __esimd_lrp<SZ>(src0.data(), Src1.data(), Src2.data());

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<float>(Result);
}

#else

// The old implementation is to generate vISA IRs for lrp.
// Now We change to use direct mul/add, and hope to generate mad instructions
// at the end, to still get the performance as good as HW solution.
// The only input and return types for these APIs are floats.
// In order to be able to use the old emu code, we keep the template argument
// for the type, although the type "T" can only be float.
// We use std::enable_if to force the float type only.
// If the gen is not specified we warn the programmer that they are potentially
// using less efficient implementation.
template <typename T, int SZ, typename U, typename V,
          class Sat = __ESIMD_NS::saturation_off_tag>
__SYCL_DEPRECATED("Gen9 specific: use emulation sequence")
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T>::value &&
                         std::is_floating_point<T>::value &&
                         __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
                         std::is_floating_point<U>::value,
                     __ESIMD_NS::simd<T, SZ>> lrp(__ESIMD_NS::simd<T, SZ> src0,
                                                  U src1, V src2,
                                                  Sat sat = {}) {

  __ESIMD_NS::simd<float, SZ> Src1 = src1;
  __ESIMD_NS::simd<float, SZ> Src2 = src2;
  __ESIMD_NS::simd<float, SZ> Result;
  Result = Src1 * src0 + Src2 * (1.0f - src0);
  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T>(Result);
}
#endif

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
  __ESIMD_NS::simd<float, N> v_distance;
  __ESIMD_NS::simd<float, N> atan2;
  __ESIMD_NS::simd_mask<N> mask;

  constexpr float CONST_DBL_EPSILON = 0.00001f;

  mask = (x < -CONST_DBL_EPSILON && y < CONST_DBL_EPSILON && y >= 0.f);
  atan2.merge(float(detail::__ESIMD_CONST_PI), 0.f, mask);
  mask = (x < -CONST_DBL_EPSILON && y > -CONST_DBL_EPSILON && y < 0);
  atan2.merge(float(-detail::__ESIMD_CONST_PI), mask);
  mask = (x < CONST_DBL_EPSILON && __ESIMD_NS::abs(y) > CONST_DBL_EPSILON);
  v_distance = __ESIMD_NS::sqrt(x * x + y * y);
  atan2.merge(2.0f * esimd::atan((v_distance - x) / y), mask);

  mask = (x > 0.f);
  atan2.merge(2.0f * esimd::atan(y / (v_distance + x)), mask);

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

/// This enum is used to encode all possible logical operations performed
/// on the 3 input operands. It is used as a template argument of the bfn()
/// function.
/// Example: d = bfn<~bfn_t::x & ~bfn_t::y & ~bfn_t::z>(s0, s1, s2);
using bfn_t __SYCL_DEPRECATED("Please use sycl::ext::intel::esimd::bfn_t") =
    __ESIMD_NS::bfn_t;

/// Performs binary function computation with three vector operands.
/// @tparam FuncControl boolean function control expressed with bfn_t
/// enum values.
/// @tparam T type of the input vector element.
/// @tparam N size of the input vector.
/// @param s0 First boolean function argument.
/// @param s1 Second boolean function argument.
/// @param s2 Third boolean function argument.
template <bfn_t FuncControl, typename T, int N>
__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::bfn<FuncControl>(src0, src1, src2);")
__ESIMD_API std::enable_if_t<std::is_integral_v<T>, __ESIMD_NS::simd<T, N>> bfn(
    __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
    __ESIMD_NS::simd<T, N> src2) {
  return __ESIMD_NS::bfn<FuncControl>(src0, src1, src2);
}

/// Performs binary function computation with three scalar operands.
/// @tparam FuncControl boolean function control expressed with bfn_t enum
/// values.
/// @tparam T type of the input vector element.
/// @param s0 First boolean function argument.
/// @param s1 Second boolean function argument.
/// @param s2 Third boolean function argument.
template <bfn_t FuncControl, typename T>
__SYCL_DEPRECATED(
    "Please use sycl::ext::intel::esimd::bfn<FuncControl>(src0, src1, src2);")
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T>::value &&
                                 std::is_integral_v<T>,
                             T> bfn(T src0, T src1, T src2) {
  return __ESIMD_NS::bfn<FuncControl>(src0, src1, src2);
}

/// rdtsc - get the value of timestamp counter.
/// \return the current value of timestamp counter
ESIMD_INLINE uint64_t rdtsc() {
  __ESIMD_NS::simd<uint32_t, 4> retv = __esimd_timestamp();
  return retv.template bit_cast_view<uint64_t>()[0];
}

/// @} sycl_esimd_logical

} // namespace ext::intel::experimental::esimd
} // namespace _V1
} // namespace sycl
