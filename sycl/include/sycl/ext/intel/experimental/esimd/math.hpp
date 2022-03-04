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

#include <sycl/ext/intel/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/math_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

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
  using ComputationTy = __ESIMD_DNS::computation_type_t<decltype(src0), U>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_on_tag>) {
    if constexpr (std::is_unsigned<T0>::value) {
      if constexpr (std::is_unsigned<T1>::value)
        return __esimd_uushl_sat<T0, T1, SZ>(Src0.data(), Src1.data());
      else
        return __esimd_usshl_sat<T0, T1, SZ>(Src0.data(), Src1.data());
    } else {
      if constexpr (std::is_signed<T1>::value)
        return __esimd_sushl_sat<T0, T1, SZ>(Src0.data(), Src1.data());
      else
        return __esimd_ssshl_sat<T0, T1, SZ>(Src0.data(), Src1.data());
    }
  } else {
    if constexpr (std::is_unsigned<T0>::value) {
      if constexpr (std::is_unsigned<T1>::value)
        return __esimd_uushl<T0, T1, SZ>(Src0.data(), Src1.data());
      else
        return __esimd_usshl<T0, T1, SZ>(Src0.data(), Src1.data());
    } else {
      if constexpr (std::is_signed<T1>::value)
        return __esimd_sushl<T0, T1, SZ>(Src0.data(), Src1.data());
      else
        return __esimd_ssshl<T0, T1, SZ>(Src0.data(), Src1.data());
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
  using ComputationTy = __ESIMD_DNS::computation_type_t<T1, T2>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  __ESIMD_NS::simd<T0, 1> Result = esimd::shl<T0>(Src0, Src1, sat);
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
  using ComputationTy = __ESIMD_DNS::computation_type_t<decltype(src0), U>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  // TODO H/W supports saturation with this op - map to more efficient version.
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Result =
      Src0.data() >> Src1.data();

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return Result;
  else
    return __ESIMD_NS::saturate<T0>(Result);
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
  using ComputationTy = __ESIMD_DNS::computation_type_t<T1, T2>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  __ESIMD_NS::simd<T0, 1> Result = esimd::shr<T0>(Src0, Src1, sat);
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
__ESIMD_API
    std::enable_if_t<std::is_integral<T0>::value && std::is_integral<T1>::value,
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
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             __ESIMD_NS::simd<T0, SZ>>
rol(__ESIMD_NS::simd<T1, SZ> src0, U src1) {
  using ComputationTy = __ESIMD_DNS::computation_type_t<decltype(src0), U>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  return __esimd_rol<T0>(Src0.data(), Src1.data());
}

/// Rotate left operation with two scalar inputs
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return rotated left value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T2>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<T2>::value,
                             std::remove_const_t<T0>>
rol(T1 src0, T2 src1) {
  using ComputationTy = __ESIMD_DNS::computation_type_t<T1, T2>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  __ESIMD_NS::simd<T0, 1> Result = esimd::rol<T0>(Src0, Src1);
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
__ESIMD_API
    std::enable_if_t<std::is_integral<T0>::value && std::is_integral<T1>::value,
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
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             __ESIMD_NS::simd<T0, SZ>>
ror(__ESIMD_NS::simd<T1, SZ> src0, U src1) {
  using ComputationTy = __ESIMD_DNS::computation_type_t<decltype(src0), U>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  return __esimd_ror<T0>(Src0.data(), Src1.data());
}

/// Rotate right operation with two scalar inputs
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return rotated right value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T2>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<T2>::value,
                             std::remove_const_t<T0>>
ror(T1 src0, T2 src1) {
  using ComputationTy = __ESIMD_DNS::computation_type_t<T1, T2>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  __ESIMD_NS::simd<T0, 1> Result = esimd::ror<T0>(Src0, Src1);
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
  // TODO H/W supports saturation with this op - map to more efficient version.
  __ESIMD_NS::simd<ComputationTy, SZ> Result = Src0.data() >> src1.data();

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
  using ComputationTy = __ESIMD_DNS::computation_type_t<T1, T2>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  __ESIMD_NS::simd<T0, 1> Result = esimd::lsr<T0>(Src0, Src1, sat);
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
  using ComputationTy = __ESIMD_DNS::computation_type_t<T1, T2>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  __ESIMD_NS::simd<T0, 1> Result = esimd::asr<T0>(Src0, Src1, sat);
  return Result[0];
}
/// @} sycl_esimd_bitmanip

/// @addtogroup sycl_esimd_math
/// @{

// imul
#ifndef ESIMD_HAS_LONG_LONG
// use mulh instruction for high half
template <typename T0, typename T1, typename U, int SZ>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_dword_type<T0>::value &&
                                      __ESIMD_DNS::is_dword_type<T1>::value &&
                                      __ESIMD_DNS::is_dword_type<U>::value,
                                  __ESIMD_NS::simd<T0, SZ>>
    imul(__ESIMD_NS::simd<T0, SZ> &rmd, __ESIMD_NS::simd<T1, SZ> src0, U src1) {
  using ComputationTy = __ESIMD_DNS::computation_type_t<decltype(src0), U>;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src0 = src0;
  typename __ESIMD_DNS::simd_type<ComputationTy>::type Src1 = src1;
  rmd = Src0 * Src1;
  if constexpr (std::is_unsigned<T0>::value)
    return __esimd_umulh(Src0.data(), Src1.data());
  else
    return __esimd_smulh(Src0.data(), Src1.data());
}

#else
// imul bdw+ version: use qw=dw*dw multiply.
// We need to special case SZ==1 to avoid "error: when select size is 1, the
// stride must also be 1" on the selects.
template <typename T0, typename T1, typename U, int SZ>
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_dword_type<T0>::value &&
                         __ESIMD_DNS::is_dword_type<T1>::value &&
                         __ESIMD_DNS::is_dword_type<U>::value && SZ == 1,
                     __ESIMD_NS::simd<T0, SZ>>
    imul(__ESIMD_NS::simd<T0, SZ> &rmd, __ESIMD_NS::simd<T1, SZ> src0, U src1) {
  using ComputationTy =
      __ESIMD_DNS::computation_type_t<decltype(rmd), long long>;
  ComputationTy Product = convert<long long>(src0);
  Product *= src1;
  rmd = Product.bit_cast_view<T0>().select<1, 1>[0];
  return Product.bit_cast_view<T0>().select<1, 1>[1];
}

template <typename T0, typename T1, typename U, int SZ>
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_dword_type<T0>::value &&
                         __ESIMD_DNS::is_dword_type<T1>::value &&
                         __ESIMD_DNS::is_dword_type<U>::value && SZ != 1,
                     __ESIMD_NS::simd<T0, SZ>>
    imul(__ESIMD_NS::simd<T0, SZ> &rmd, __ESIMD_NS::simd<T1, SZ> src0, U src1) {
  using ComputationTy =
      __ESIMD_DNS::computation_type_t<decltype(rmd), long long>;
  ComputationTy Product = convert<long long>(src0);
  Product *= src1;
  rmd = Product.bit_cast_view<T0>().select<SZ, 2>(0);
  return Product.bit_cast_view<T0>().select<SZ, 2>(1);
}
#endif

// TODO: document
template <typename T0, typename T1, typename U, int SZ>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<U>::value,
                             __ESIMD_NS::simd<T0, SZ>>
imul(__ESIMD_NS::simd<T0, SZ> &rmd, U src0, __ESIMD_NS::simd<T1, SZ> src1) {
  return esimd::imul(rmd, src1, src0);
}

// TODO: document
template <typename T0, typename T, typename U>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T>::value &&
                                      __ESIMD_DNS::is_esimd_scalar<U>::value &&
                                      __ESIMD_DNS::is_esimd_scalar<T0>::value,
                                  T0>
    imul(__ESIMD_NS::simd<T0, 1> &rmd, T src0, U src1) {
  __ESIMD_NS::simd<T, 1> src_0 = src0;
  __ESIMD_NS::simd<U, 1> src_1 = src1;
  __ESIMD_NS::simd<T0, 1> res =
      esimd::imul(rmd, src_0.select_all(), src_1.select_all());
  return res[0];
}

/// Integral quotient (vector version)
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1.
/// @param src0 the dividend input vector.
/// @param src1 the divisor scalar value.
/// @return vector of quotient elements.
template <typename T, int SZ, typename U>
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                     __ESIMD_NS::simd<T, SZ>>
    quot(__ESIMD_NS::simd<T, SZ> src0, U src1) {
  return src0 / src1;
}

/// Integral quotient (scalar version)
/// @tparam T0 element type of the dividend \p src0 and returned value.
/// @tparam T1 element type of the divisor \p src1.
/// @param src0 the dividend.
/// @param src1 the divisor.
/// @return quotient value.
template <typename T0, typename T1>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value,
                             std::remove_const_t<T0>>
quot(T0 src0, T1 src1) {
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
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                     __ESIMD_NS::simd<T, SZ>>
    mod(__ESIMD_NS::simd<T, SZ> src0, U src1) {
  return src0 % src1;
}

/// Modulo (scalar version)
/// @tparam T0 element type of the dividend \p src0 and returned value.
/// @tparam T1 element type of the divisor \p src1.
/// @param src0 the dividend.
/// @param src1 the divisor.
/// @return Modulo value.
template <typename T0, typename T1>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                 __ESIMD_DNS::is_esimd_scalar<T1>::value &&
                                 std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value,
                             std::remove_const_t<T0>>
mod(T0 src0, T1 src1) {
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
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                     __ESIMD_NS::simd<T, SZ>>
    div(__ESIMD_NS::simd<T, SZ> &remainder, __ESIMD_NS::simd<T, SZ> src0,
        U src1) {
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
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value &&
                         __ESIMD_DNS::is_esimd_scalar<U>::value,
                     __ESIMD_NS::simd<T, SZ>>
    div(__ESIMD_NS::simd<T, SZ> &remainder, U src0,
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
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<RT>::value &&
                                      __ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                      __ESIMD_DNS::is_esimd_scalar<T1>::value,
                                  std::remove_const_t<RT>>
    div(__ESIMD_NS::simd<std::remove_const_t<RT>, 1> &remainder, T0 src0,
        T1 src1) {
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
__ESIMD_API __ESIMD_NS::simd<RT, SZ>
line(float P, float Q, __ESIMD_NS::simd<T, SZ> src1, Sat sat = {}) {
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
// We use enable_if to force the float type only.
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
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T1>::value &&
                         std::is_floating_point<T1>::value &&
                         __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
                         std::is_floating_point<U>::value,
                     __ESIMD_NS::simd<T0, SZ>>
    dp2(__ESIMD_NS::simd<T1, SZ> src0, U src1, Sat sat = {}) {
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
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T1>::value &&
                         std::is_floating_point<T1>::value &&
                         __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
                         std::is_floating_point<U>::value,
                     __ESIMD_NS::simd<T0, SZ>>
    dp3(__ESIMD_NS::simd<T1, SZ> src0, U src1, Sat sat = {}) {
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
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T1>::value &&
                         std::is_floating_point<T1>::value &&
                         __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
                         std::is_floating_point<U>::value,
                     __ESIMD_NS::simd<T0, SZ>>
    dp4(__ESIMD_NS::simd<T1, SZ> src0, U src1, Sat sat = {}) {
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
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T>::value &&
                         std::is_floating_point<T>::value &&
                         __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
                         std::is_floating_point<U>::value,
                     __ESIMD_NS::simd<T, SZ>>
    dph(__ESIMD_NS::simd<T, SZ> src0, U src1, Sat sat = {}) {
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
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T>::value &&
                                      std::is_floating_point<T>::value,
                                  __ESIMD_NS::simd<T, SZ>>
    line(__ESIMD_NS::simd<T, 4> src0, __ESIMD_NS::simd<T, SZ> src1,
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
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T>::value &&
                                      std::is_floating_point<T>::value,
                                  __ESIMD_NS::simd<T, SZ>>
    line(float P, float Q, __ESIMD_NS::simd<T, SZ> src1, Sat sat = {}) {
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

// lzd
template <typename RT, typename T0, int SZ,
          class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<RT, SZ> lzd(__ESIMD_NS::simd<T0, SZ> src0,
                                         Sat sat = {}) {
  // Saturation parameter ignored
  __ESIMD_NS::simd<uint, SZ> Src0 = src0;
  return __esimd_lzd<uint>(Src0.data());
}

template <typename RT, typename T0, class Sat = __ESIMD_NS::saturation_off_tag>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<RT>::value &&
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
// We use enable_if to force the float type only.
// If the gen is not specified we warn the programmer that they are potentially
// using less efficient implementation.
template <typename T, int SZ, typename U, typename V,
          class Sat = __ESIMD_NS::saturation_off_tag>
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<__ESIMD_DNS::is_fp_or_dword_type<T>::value &&
                         std::is_floating_point<T>::value &&
                         __ESIMD_DNS::is_fp_or_dword_type<U>::value &&
                         std::is_floating_point<U>::value,
                     __ESIMD_NS::simd<T, SZ>>
    lrp(__ESIMD_NS::simd<T, SZ> src0, U src1, V src2, Sat sat = {}) {

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
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                      __ESIMD_DNS::is_esimd_scalar<T1>::value,
                                  std::remove_const_t<T0>>
    bf_reverse(T1 src0) {
  __ESIMD_NS::simd<T1, 1> Src0 = src0;
  __ESIMD_NS::simd<T0, 1> Result = esimd::bf_reverse<T0>(Src0);
  return Result[0];
}

/// bf_insert
template <typename T0, typename T1, int SZ, typename U, typename V, typename W>
ESIMD_NODEBUG ESIMD_INLINE
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
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
                                      __ESIMD_DNS::is_esimd_scalar<T4>::value,
                                  std::remove_const_t<T0>>
    bf_insert(T1 src0, T2 src1, T3 src2, T4 src3) {
  __ESIMD_NS::simd<T4, 1> Src3 = src3;
  __ESIMD_NS::simd<T0, 1> Result = esimd::bf_insert<T0>(src0, src1, src2, Src3);
  return Result[0];
}

/// bf_extract
template <typename T0, typename T1, int SZ, typename U, typename V>
ESIMD_NODEBUG ESIMD_INLINE
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
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<__ESIMD_DNS::is_esimd_scalar<T0>::value &&
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
constexpr double HDR_CONST_PI = 3.1415926535897932384626433832795;
} // namespace detail
/// @endcond ESIMD_DETAIL

template <typename T, int SZ>
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<std::is_floating_point<T>::value, __ESIMD_NS::simd<T, SZ>>
    atan(__ESIMD_NS::simd<T, SZ> src0) {
  __ESIMD_NS::simd<T, SZ> Src0 = __ESIMD_NS::abs(src0);

  __ESIMD_NS::simd_mask<SZ> Neg = src0 < T(0.0);
  __ESIMD_NS::simd_mask<SZ> Gt1 = Src0 > T(1.0);

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

  Result.merge(Result - T(detail::HDR_CONST_PI / 2.0), Gt1);
  Result.merge(Result, Neg);
  return Result;
}

template <typename T>
__ESIMD_API std::enable_if_t<std::is_floating_point<T>::value, T> atan(T src0) {
  __ESIMD_NS::simd<T, 1> Src0 = src0;
  __ESIMD_NS::simd<T, 1> Result = esimd::atan(Src0);
  return Result[0];
}

// acos

template <typename T, int SZ>
ESIMD_NODEBUG ESIMD_INLINE
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
  Result.merge(T(detail::HDR_CONST_PI) - Result, Neg);
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
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<std::is_floating_point<T>::value, __ESIMD_NS::simd<T, SZ>>
    asin(__ESIMD_NS::simd<T, SZ> src0) {
  __ESIMD_NS::simd_mask<SZ> Neg = src0 < T(0.0);

  __ESIMD_NS::simd<T, SZ> Result =
      T(detail::HDR_CONST_PI / 2.0) - esimd::acos(__ESIMD_NS::abs(src0));

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

/// @addtogroup sycl_esimd_conv
/// @{

////////////////////////////////////////////////////////////////////////////////
// Rounding intrinsics.
////////////////////////////////////////////////////////////////////////////////

#define __ESIMD_INTRINSIC_DEF(name)                                            \
  /** @tparam T Element type.                                               */ \
  /** @tparam SZ Number of elements in the input vector.                    */ \
  /** @tparam Sat Saturation control. Default is \c                            \
   * __ESIMD_NS::saturation_off_tag      */                                    \
  /** @param src0 The argument to perform rounding on.                      */ \
  /** @param sat The type tag object to auto-deduce saturation control.     */ \
  /**   can be \c saturation_off or \c saturation_on                        */ \
  template <typename T, int SZ, class Sat = __ESIMD_NS::saturation_off_tag>    \
  __ESIMD_API __ESIMD_NS::simd<T, SZ> name(__ESIMD_NS::simd<float, SZ> src0,   \
                                           Sat sat = {}) {                     \
    __ESIMD_NS::simd<float, SZ> Result = __esimd_##name<SZ>(src0.data());      \
    if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)         \
      return Result;                                                           \
    else if constexpr (!std::is_same_v<float, T>) {                            \
      auto RawRes = __ESIMD_NS::saturate<float>(Result).data();                \
      return __ESIMD_DNS::convert_vector<T, float, SZ>(std::move(RawRes));     \
    } else {                                                                   \
      return __ESIMD_NS::saturate<T>(Result);                                  \
    }                                                                          \
  }                                                                            \
  /** Scalar version.                                                       */ \
  template <typename T, class Sat = __ESIMD_NS::saturation_off_tag>            \
  __ESIMD_API T name(float src0, Sat sat = {}) {                               \
    __ESIMD_NS::simd<float, 1> Src0 = src0;                                    \
    __ESIMD_NS::simd<T, 1> Result = name<T>(Src0, sat);                        \
    return Result[0];                                                          \
  }

/// Round-down (also known as \c floor). Supports only \c float.
/// Corner cases:
/// | _        | _    | _       | _  | _  | _       | _    | _
/// |----------|------|---------|----|----|---------|------|----
/// | **src0** | -inf | -denorm | -0 | +0 | +denorm | +inf | NaN
/// | **dst**  | -inf | \*      | -0 | +0 | +0      | +inf | NaN
/// - \* \c -1 or \c -0 depending on the Single Precision Denorm Mode.
__ESIMD_INTRINSIC_DEF(rndd)

/// Round-up (also known as \c ceil). Supports only \c float.
/// Corner cases:
/// | _        | _    | _       | _  | _  | _       | _    | _
/// |----------|------|---------|----|----|---------|------|----
/// | **src0** | -inf | -denorm | -0 | +0 | +denorm | +inf | NaN
/// | **dst**  | -inf | -0      | -0 | +0 | \*      | +inf | NaN
/// - \* \c +1 or \c +0 depending on the Single Precision Denorm Mode.
__ESIMD_INTRINSIC_DEF(rndu)

/// Round-to-even (also known as \c round). Supports only \c float.
/// Corner cases:
/// | _        | _    | _       | _  | _  | _       | _    | _
/// |----------|------|---------|----|----|---------|------|----
/// | **src0** | -inf | -denorm | -0 | +0 | +denorm | +inf | NaN
/// | **dst**  | -inf | -0      | -0 | +0 | +0      | +inf | NaN
__ESIMD_INTRINSIC_DEF(rnde)

/// Round-to-zero (also known as \c trunc). Supports only \c float.
/// Corner cases:
/// | _        | _    | _       | _  | _  | _       | _    | _
/// |----------|------|---------|----|----|---------|------|----
/// | **src0** | -inf | -denorm | -0 | +0 | +denorm | +inf | NaN
/// | **dst**  | -inf | -0      | -0 | +0 | +0      | +inf | NaN
__ESIMD_INTRINSIC_DEF(rndz)

#undef __ESIMD_INTRINSIC_DEF
/// @} sycl_esimd_conv

/// @addtogroup sycl_esimd_conv
/// @{

/// "Floor" operation, vector version - alias of \c rndd.
template <typename RT, int SZ, class Sat = __ESIMD_NS::saturation_off_tag>
ESIMD_INLINE __ESIMD_NS::simd<RT, SZ>
floor(const __ESIMD_NS::simd<float, SZ> src0, Sat sat = {}) {
  return esimd::rndd<RT, SZ>(src0, sat);
}

/// "Floor" operation, scalar version - alias of \c rndd.
template <typename RT, class Sat = __ESIMD_NS::saturation_off_tag>
ESIMD_INLINE RT floor(float src0, Sat sat = {}) {
  return esimd::rndd<RT, 1U>(src0, sat)[0];
}

/// "Ceiling" operation, vector version - alias of \c rndu.
template <typename RT, int SZ, class Sat = __ESIMD_NS::saturation_off_tag>
ESIMD_INLINE __ESIMD_NS::simd<RT, SZ>
ceil(const __ESIMD_NS::simd<float, SZ> src0, Sat sat = {}) {
  return esimd::rndu<RT, SZ>(src0, sat);
}

/// "Ceiling" operation, scalar version - alias of \c rndu.
template <typename RT, class Sat = __ESIMD_NS::saturation_off_tag>
ESIMD_INLINE RT ceil(float src0, Sat sat = {}) {
  return esimd::rndu<RT, 1U>(src0, sat);
}

/// Round to integral value using the round to zero rounding mode (vector
/// version). Alias of \c rndz.
/// @tparam RT element type of the return vector.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of rounded values.
template <typename RT, int SZ, class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<RT, SZ>
trunc(const __ESIMD_NS::simd<float, SZ> &src0, Sat sat = {}) {
  return esimd::rndz<RT, SZ>(src0, sat);
}

/// Round to integral value using the round to zero rounding mode (scalar
/// version). Alias of \c rndz.
/// @tparam RT type of the return value.
/// @param src0 the input operand.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return rounded value.
template <typename RT, class Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API RT trunc(float src0, Sat sat = {}) {
  return esimd::rndz<RT, 1U>(src0, sat)[0];
}

/// @} sycl_esimd_conv

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

/// @cond ESIMD_DETAIL

namespace detail {
static auto constexpr CONST_PI = 3.14159f;
static auto constexpr CMPI = 3.14159265f;
} // namespace detail

/// @endcond ESIMD_DETAIL

// For vector input
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N>
atan2_fast(__ESIMD_NS::simd<float, N> y, __ESIMD_NS::simd<float, N> x) {
  __ESIMD_NS::simd<float, N> a0;
  __ESIMD_NS::simd<float, N> a1;
  __ESIMD_NS::simd<float, N> atan2;

  __ESIMD_NS::simd_mask<N> mask = (y >= 0.0f);
  a0.merge(detail::CONST_PI * 0.5f, detail::CONST_PI * 1.5f, mask);
  a1.merge(0, detail::CONST_PI * 2.0f, mask);

  a1.merge(detail::CONST_PI, x < 0.0f);

  __ESIMD_NS::simd<float, N> xy = x * y;
  __ESIMD_NS::simd<float, N> x2 = x * x;
  __ESIMD_NS::simd<float, N> y2 = y * y;

  /* smallest such that 1.0+CONST_DBL_EPSILON != 1.0 */
  constexpr auto CONST_DBL_EPSILON = 0.00001f;

  a0 -= (xy / (y2 + x2 * 0.28f + CONST_DBL_EPSILON));
  a1 += (xy / (x2 + y2 * 0.28f + CONST_DBL_EPSILON));

  atan2.merge(a1, a0, y2 <= x2);
  return atan2;
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
  __ESIMD_NS::simd<float, N> v_y0;
  __ESIMD_NS::simd<float, N> atan2;
  __ESIMD_NS::simd_mask<N> mask;

  mask = (x < 0);
  v_y0.merge(detail::CONST_PI, 0, mask);
  v_distance = __ESIMD_NS::sqrt(x * x + y * y);
  mask = (__ESIMD_NS::abs<float>(y) < 0.000001f);
  atan2.merge(v_y0, (2 * esimd::atan((v_distance - x) / y)), mask);
  return atan2;
}

// For Scalar Input
template <> ESIMD_INLINE float atan2(float y, float x) {
  float v_distance;
  float v_y0;
  __ESIMD_NS::simd<float, 1> atan2;
  __ESIMD_NS::simd_mask<1> mask;

  mask = (x < 0);
  v_y0 = mask[0] ? detail::CONST_PI : 0;
  v_distance = __ESIMD_NS::sqrt<float>(x * x + y * y);
  mask = (__ESIMD_NS::abs<float>(y) < 0.000001f);
  atan2.merge(v_y0, (2 * esimd::atan((v_distance - x) / y)), mask);
  return atan2[0];
}

// fmod:
// For Vector input
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N> fmod(__ESIMD_NS::simd<float, N> y,
                                             __ESIMD_NS::simd<float, N> x) {
  __ESIMD_NS::simd<int, N> v_quot;
  __ESIMD_NS::simd<float, N> fmod;

  v_quot = convert<int>(y / x);
  fmod = y - x * convert<float>(v_quot);
  return fmod;
}

//     For Scalar Input
template <> ESIMD_INLINE float fmod(float y, float x) {
  int v_quot;
  __ESIMD_NS::simd<float, 1> fmod;

  v_quot = (int)(y / x);
  fmod = y - x * v_quot;
  return fmod[0];
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
  __ESIMD_NS::simd<float, N> TwoPI(6.2831853f);
  __ESIMD_NS::simd<float, N> CmpI(detail::CMPI);
  __ESIMD_NS::simd<float, N> OneP(1.f);
  __ESIMD_NS::simd<float, N> OneN(-1.f);

  x = esimd::fmod(x, TwoPI);

  x1.merge(CmpI - x, x - CmpI, (x <= detail::CMPI));
  x1.merge(x, (x <= detail::CMPI * 0.5f));
  x1.merge(CmpI * 2 - x, (x > detail::CMPI * 1.5f));

  sign.merge(OneN, OneP, (x > detail::CMPI));

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
template <typename T> ESIMD_INLINE float sin_emu(T x0) {
  __ESIMD_NS::simd<float, 1> x1;
  __ESIMD_NS::simd<float, 1> x2;
  __ESIMD_NS::simd<float, 1> t3;

  __ESIMD_NS::simd<float, 1> sign;
  __ESIMD_NS::simd<float, 1> fTrig;
  float TwoPI = detail::CMPI * 2.0f;

  __ESIMD_NS::simd<float, 1> x = esimd::fmod(x0, TwoPI);

  __ESIMD_NS::simd<float, 1> CmpI(detail::CMPI);
  __ESIMD_NS::simd<float, 1> OneP(1.f);
  __ESIMD_NS::simd<float, 1> OneN(-1.f);

  x1.merge(CmpI - x, x - CmpI, (x <= detail::CMPI));
  x1.merge(x, (x <= detail::CMPI * 0.5f));
  x1.merge(CmpI * 2.0f - x, (x > detail::CMPI * 1.5f));

  sign.merge(OneN, OneP, (x > detail::CMPI));

  x2 = x1 * x1;
  t3 = x2 * x1 * 0.1666667f;

  fTrig =
      x1 + t3 * (OneN + x2 * 0.05f *
                            (OneP + x2 * 0.0238095f *
                                        (OneN + x2 * 0.0138889f *
                                                    (OneP - x2 * 0.0090909f))));
  fTrig *= sign;
  return fTrig[0];
}

// cos_emu - EU emulation for sin(x)
// For Vector input
template <int N>
ESIMD_INLINE __ESIMD_NS::simd<float, N> cos_emu(__ESIMD_NS::simd<float, N> x) {
  __ESIMD_NS::simd<float, N> x1;
  __ESIMD_NS::simd<float, N> x2;
  __ESIMD_NS::simd<float, N> t2;
  __ESIMD_NS::simd<float, N> t3;

  __ESIMD_NS::simd<float, N> sign;
  __ESIMD_NS::simd<float, N> fTrig;
  __ESIMD_NS::simd<float, N> TwoPI(6.2831853f);
  __ESIMD_NS::simd<float, N> CmpI(detail::CMPI);
  __ESIMD_NS::simd<float, N> OneP(1.f);
  __ESIMD_NS::simd<float, N> OneN(-1.f);

  x = esimd::fmod(x, TwoPI);

  x1.merge(x - detail::CMPI * 0.5f, CmpI * 1.5f - x, (x <= detail::CMPI));
  x1.merge(CmpI * 0.5f - x, (x <= detail::CMPI * 0.5f));
  x1.merge(x - detail::CMPI * 1.5f, (x > detail::CMPI * 1.5f));

  sign.merge(1, -1, ((x < detail::CMPI * 0.5f) | (x >= detail::CMPI * 1.5f)));

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
template <typename T> ESIMD_INLINE float cos_emu(T x0) {
  __ESIMD_NS::simd<float, 1> x1;
  __ESIMD_NS::simd<float, 1> x2;
  __ESIMD_NS::simd<float, 1> t3;

  __ESIMD_NS::simd<float, 1> sign;
  __ESIMD_NS::simd<float, 1> fTrig;
  float TwoPI = detail::CMPI * 2.0f;

  __ESIMD_NS::simd<float, 1> x = esimd::fmod(x0, TwoPI);

  __ESIMD_NS::simd<float, 1> CmpI(detail::CMPI);
  __ESIMD_NS::simd<float, 1> OneP(1.f);
  __ESIMD_NS::simd<float, 1> OneN(-1.f);

  x1.merge(x - detail::CMPI * 0.5f, CmpI * 1.5f - x, (x <= detail::CMPI));
  x1.merge(CmpI * 0.5f - x, (x <= detail::CMPI * 0.5f));
  x1.merge(x - detail::CMPI * 1.5f, (x > detail::CMPI * 1.5f));

  sign.merge(OneP, OneN,
             ((x < detail::CMPI * 0.5f) | (x >= detail::CMPI * 1.5f)));

  x2 = x1 * x1;
  t3 = x2 * x1 * 0.1666667f;
  fTrig =
      x1 + t3 * (OneN + x2 * 0.05f *
                            (OneP + x2 * 0.0238095f *
                                        (OneN + x2 * 0.0138889f *
                                                    (OneP - x2 * 0.0090909f))));
  fTrig *= sign;
  return fTrig[0];
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
  constexpr float log2E = 1.442695f; // same as esimd::log(e)

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
  auto temp = __ESIMD_NS::exp(absX * 2.0f * log2E) + 1.f;
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
  constexpr float xlarge = 88.f;
  constexpr float log2E = 1.442695f; // same as esimd::log(e)

  using RT = __ESIMD_NS::simd<float, N>;

  RT absX = __ESIMD_NS::abs(x);

  RT sign;
  sign.merge(-1.f, 1.f, x < 0.f);

  auto isLarge = (absX > xlarge);
  auto isLessE = (absX <= xlarge);

  RT res;
  res.merge(sign, x, isLarge);

  RT exp;
  exp = __ESIMD_NS::exp(absX * 2.f * log2E);

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

/// @} sycl_esimd_math

/// @cond ESIMD_DETAIL
// dpas helpers
namespace detail {

enum class dpas_ops_per_channel : unsigned {
  OP1 = 1u,
  OP2 = 2u,
  OP4 = 4u,
  OP8 = 8u,
  INVALID = 0xffffffffu
};
constexpr dpas_ops_per_channel
get_ops_per_channel(argument_type src1_precision,
                    argument_type src2_precision) {
  if ((src1_precision == argument_type::U8) ||
      (src1_precision == argument_type::S8)) {
    if ((src2_precision == argument_type::U8) ||
        (src2_precision == argument_type::S8) ||
        (src2_precision == argument_type::U4) ||
        (src2_precision == argument_type::S4) ||
        (src2_precision == argument_type::U2) ||
        (src2_precision == argument_type::S2)) {
      return dpas_ops_per_channel::OP4;
    }
  } else if ((src1_precision == argument_type::U4) ||
             (src1_precision == argument_type::S4) ||
             (src1_precision == argument_type::U2) ||
             (src1_precision == argument_type::S2)) {
    if ((src2_precision == argument_type::U8) ||
        (src2_precision == argument_type::S8)) {
      return dpas_ops_per_channel::OP4;
    } else if ((src2_precision == argument_type::U4) ||
               (src2_precision == argument_type::S4) ||
               (src2_precision == argument_type::U2) ||
               (src2_precision == argument_type::S2)) {
      return dpas_ops_per_channel::OP8;
    }
  } else if ((src1_precision == argument_type::BF16) &&
             (src2_precision == argument_type::BF16)) {
    return dpas_ops_per_channel::OP2;
  } else if ((src1_precision == argument_type::FP16) &&
             (src2_precision == argument_type::FP16)) {
    return dpas_ops_per_channel::OP2;
  } else if ((src1_precision == argument_type::TF32) &&
             (src2_precision == argument_type::TF32)) {
    return dpas_ops_per_channel::OP1;
  }
  return dpas_ops_per_channel::INVALID;
}

constexpr unsigned get_precision_bits(argument_type src_precision) {
  if ((src_precision == argument_type::U8) ||
      (src_precision == argument_type::S8)) {
    return 8;
  } else if ((src_precision == argument_type::U4) ||
             (src_precision == argument_type::S4)) {
    return 4;
  } else if ((src_precision == argument_type::U2) ||
             (src_precision == argument_type::S2)) {
    return 2;
  } else if ((src_precision == argument_type::BF16) ||
             (src_precision == argument_type::FP16)) {
    return 16;
  } else if (src_precision == argument_type::TF32) {
    return 32;
  }
  return 0;
}

} // namespace detail
/// @endcond ESIMD_DETAIL

/// @defgroup sycl_esimd_systolic_array_api Systolic Array APIs.
/// APIs below are used to implement dot product accumulate systolic functions
/// @ingroup sycl_esimd

/// @addtogroup sycl_esimd_systolic_array_api
/// @{
/// DPAS
/// @param src0 is the source operand that represents accumulator for the dpas
/// function
/// @param src1 is the first source perand with data precision type specified
/// by src1_precision.
/// @param src2 is the second source operand with data precision type specified
/// by src2_precision.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return the vector value of DPAS computation result.
template <argument_type src1_precision, argument_type src2_precision,
          typename T, int systolic_depth, int repeat_count, typename T0,
          typename T1, typename T2, int N, int N1, int N2,
          typename Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<T, N>
dpas(__ESIMD_NS::simd<T0, N> src0, __ESIMD_NS::simd<T1, N1> src1,
     __ESIMD_NS::simd<T2, N2> src2, Sat sat = {}) {
  // types: dst, src0, src1, src2
  // ud, d | ud, d | ub, b | ub, b
  // ud, d | ud, d | u4, s4, u2, s2 | ub, b
  // ud, d | ud, d | ub, b | u4, s4, u2, s2
  // ud, d | ud, d | u4, s4, u2, s2 | u4, s4, u2, s2
  constexpr bool check_integer =
      detail::is_one_of_v<T, unsigned int, int> &&
      detail::is_one_of_v<T0, unsigned int, int> &&
      detail::is_one_of_enum_v<argument_type, src1_precision, argument_type::S8,
                               argument_type::U8, argument_type::U4,
                               argument_type::S4, argument_type::U2,
                               argument_type::S2> &&
      detail::is_one_of_enum_v<argument_type, src2_precision, argument_type::S8,
                               argument_type::U8, argument_type::U4,
                               argument_type::S4, argument_type::U2,
                               argument_type::S2>;
  // f, bf | f, bf | bf | bf
  constexpr bool check_bf16 =
      detail::is_one_of_v<T, float, short> &&
      detail::is_one_of_v<T0, float, short> &&
      detail::is_one_of_enum_v<argument_type, src1_precision,
                               argument_type::BF16> &&
      detail::is_one_of_enum_v<argument_type, src2_precision,
                               argument_type::BF16>;
  // f,hf | f, hf | hf | hf
  constexpr bool check_hf =
      detail::is_one_of_v<T, float, half> &&
      detail::is_one_of_v<T0, float, half> &&
      detail::is_one_of_enum_v<argument_type, src1_precision,
                               argument_type::FP16> &&
      detail::is_one_of_enum_v<argument_type, src2_precision,
                               argument_type::FP16>;

#if defined(ESIMD_XE_HPC) || defined(ESIMD_XE_HPG)
  // f | f | tf32 | tf32
  constexpr bool check_tf32 =
      detail::is_one_of_v<T, float> && detail::is_one_of_v<T0, float> &&
      detail::is_one_of_enum_v<argument_type, src1_precision,
                               argument_type::TF32> &&
      detail::is_one_of_enum_v<argument_type, src2_precision,
                               argument_type::TF32>;
#endif // defined(ESIMD_XE_HPC) || defined(ESIMD_XE_HPG)

#if defined(ESIMD_XE_HPC) || defined(ESIMD_XE_HPG)
  constexpr bool check_passed =
      (check_integer || check_hf || check_bf16 || check_tf32);
  static_assert(check_passed,
                "unsupported dpas type! The supported types are:\n"
                "    dst    |    src0    |      src1      |      src2      \n"
                "   ud, d   |   ud, d    |     ub, b      |     ub, b      \n"
                "   ud, d   |   ud, d    | u4, s4, u2, s2 | u4, s4, u2, s2 \n"
                "   f, bf   |    f, bf   |       bf       |       bf       \n"
                "   f, hf   |    f, hf   |       hf       |       hf       \n"
                "    f      |     f      |      tf32      |      tf32      \n");
#else  // else defined(ESIMD_XE_HPC) || defined(ESIMD_XE_HPG)
  constexpr bool check_passed = (check_integer || check_hf || check_bf16);
  static_assert(check_passed,
                "unsupported dpas type! The supported types are:\n"
                "    dst    |    src0    |      src1      |      src2      \n"
                "   ud, d   |   ud, d    |     ub, b      |     ub, b      \n"
                "   ud, d   |   ud, d    | u4, s4, u2, s2 | u4, s4, u2, s2 \n"
                "   f, bf   |    f, bf   |       bf       |       bf       \n"
                "   f, hf   |    f, hf   |       hf       |       hf       \n");
#endif // end else defined(ESIMD_XE_HPC) || defined(ESIMD_XE_HPG)

  static_assert(__ESIMD_DNS::is_dword_type<T1>::value,
                "Src1 must be DWORD type");
  static_assert(__ESIMD_DNS::is_dword_type<T2>::value,
                "Src2 must be DWORD type");

#if defined(ESIMD_XE_HPC) || defined(ESIMD_XE_HPG)
  static_assert((N == 16 * repeat_count), "Execution size on PVC must be 16");
#else
  static_assert((N == 8 * repeat_count), "Execution size must be 8");
#endif

  static_assert((systolic_depth == 8) || (systolic_depth == 4),
                "systolic_depth must be 8 or 4");

  static_assert((repeat_count >= 1) && (repeat_count <= 8),
                "repeat_count must be within 1 to 8");

  constexpr auto en_ops_per_channel =
      detail::get_ops_per_channel(src1_precision, src2_precision);
  static_assert(en_ops_per_channel != detail::dpas_ops_per_channel::INVALID,
                "invalid combination of Src1/Src2 precision");
  constexpr auto ops_per_channel = static_cast<unsigned>(en_ops_per_channel);

  constexpr auto src1_precision_bits =
      detail::get_precision_bits(src1_precision);
  static_assert(
      N1 == ((src1_precision_bits * systolic_depth * ops_per_channel * N) /
             (repeat_count * sizeof(T1) * 8)),
      "invalid size for Src1");

  constexpr auto src2_precision_bits =
      detail::get_precision_bits(src2_precision);
  static_assert(N2 == ((src2_precision_bits * systolic_depth * ops_per_channel *
                        repeat_count) /
                       (sizeof(T2) * 8)),
                "invalid size for Src2");

#if defined(__SYCL_DEVICE_ONLY__)
  constexpr int dst_signed = std::is_signed<T>::value;
  constexpr int src0_signed = std::is_signed<T0>::value;
  __ESIMD_NS::simd<T, N> result = __esimd_dpas<T, T0, T1, T2, N, N1, N2>(
      src0.data(), src1.data(), src2.data(), (int)src1_precision + 1,
      (int)src2_precision + 1, systolic_depth, repeat_count, dst_signed,
      src0_signed);

#else
  __ESIMD_NS::simd<T, N> result =
      __esimd_dpas<src1_precision, src2_precision, systolic_depth, repeat_count,
                   T, T0, T1, T2, N, N1, N2>(src0.data(), src1.data(),
                                             src2.data());
#endif // __SYCL_DEVICE_ONLY__

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return result;
  else
    return __ESIMD_NS::saturate<T>(result);
}

/// DPAS
/// @param src0 is the source operand that represents accumulator for the dpas
/// function, which must have the same type as return value
/// @param src1 is the first source perand with data precision type specified
/// by src1_precision.
/// @param src2 is the second source operand with data precision type specified
/// by src2_precision.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return the vector value of DPAS computation result.
template <argument_type src1_precision, argument_type src2_precision,
          int systolic_depth, int repeat_count, typename T, typename T1,
          typename T2, int N, int N1, int N2,
          typename Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<T, N>
dpas(__ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T1, N1> src1,
     __ESIMD_NS::simd<T2, N2> src2, Sat sat = {}) {
  return dpas<src1_precision, src2_precision, T, systolic_depth, repeat_count>(
      src0, src1, src2, sat);
}

/// DPAS
/// @param src1 is the first source perand with data precision type specified
/// by src1_precision.
/// @param src2 is the second source operand with data precision type specified
/// by src2_precision.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return the vector value of DPAS computation result.
template <argument_type src1_precision, argument_type src2_precision,
          int systolic_depth, int repeat_count, typename T, typename T1,
          typename T2, int N, int N1, int N2,
          typename Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<T, N> dpas(__ESIMD_NS::simd<T1, N1> src1,
                                        __ESIMD_NS::simd<T2, N2> src2,
                                        Sat sat = {}) {

  static_assert(__ESIMD_DNS::is_fp_or_dword_type<T>::value,
                "Dst must be FP or DWORD type");

  static_assert(__ESIMD_DNS::is_dword_type<T1>::value,
                "Src1 must be DWORD type");

  static_assert(__ESIMD_DNS::is_dword_type<T2>::value,
                "Src2 must be DWORD type");

  static_assert((N == 8 * repeat_count) || (N == 16 * repeat_count),
                "Execution size must be 8 or 16");

  static_assert((systolic_depth == 8) || (systolic_depth == 4),
                "systolic_depth must be 8 or 4");

  static_assert((repeat_count >= 1) && (repeat_count <= 8),
                "repeat_count must be within 1 to 8");

  constexpr auto en_ops_per_channel =
      detail::get_ops_per_channel(src1_precision, src2_precision);
  static_assert(en_ops_per_channel != detail::dpas_ops_per_channel::INVALID,
                "invalid combination of Src1/Src2 precision");
  constexpr auto ops_per_channel = static_cast<unsigned>(en_ops_per_channel);

  constexpr auto src1_precision_bits =
      detail::get_precision_bits(src1_precision);
  static_assert(
      N1 == ((src1_precision_bits * systolic_depth * ops_per_channel * N) /
             (repeat_count * sizeof(T1) * 8)),
      "invalid size for Src1");

  constexpr auto src2_precision_bits =
      detail::get_precision_bits(src2_precision);
  static_assert(N2 == ((src2_precision_bits * systolic_depth * ops_per_channel *
                        repeat_count) /
                       (sizeof(T2) * 8)),
                "invalid size for Src2");

#if defined(__SYCL_DEVICE_ONLY__)
  int dpas_info = (repeat_count << 24) + (systolic_depth << 16) +
                  (((int)src2_precision + 1) << 8) + ((int)src1_precision + 1);
  __ESIMD_NS::simd<T, N> result =
      __esimd_dpas2<T, T1, T2, N, N1, N2>(src1.data(), src2.data(), dpas_info);
#else
  __ESIMD_NS::simd<T, N> result =
      __esimd_dpas2<src1_precision, src2_precision, systolic_depth,
                    repeat_count, T, T1, T2, N, N1, N2>(src1.data(),
                                                        src2.data());
#endif // __SYCL_DEVICE_ONLY__

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return result;
  else
    return __ESIMD_NS::saturate<T>(result);
}

/// DPASW
/// @param src0 is the source operand that represents accumulator for the dpas
/// function, which must have the same type as return value.
/// @param src1 is the first source perand with data precision type specified
/// by src1_precision.
/// @param src2 is the second source operand with data precision type specified
/// by src2_precision.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return the vector value of DPAS computation result.
template <argument_type src1_precision, argument_type src2_precision,
          int systolic_depth, int repeat_count, typename T, typename T1,
          typename T2, int N, int N1, int N2,
          typename Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<T, N>
dpasw(__ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T1, N1> src1,
      __ESIMD_NS::simd<T2, N2> src2, Sat sat = {}) {
  constexpr bool is_4xhf =
      (__ESIMD_DNS::is_type<T, cl::sycl::detail::half_impl::StorageT>()) &&
      src1_precision == src2_precision && src1_precision == argument_type::FP16;

  constexpr bool is_4xbf = __ESIMD_DNS::is_word_type<T>::value &&
                           src1_precision == src2_precision &&
                           src1_precision == argument_type::BF16;

  constexpr bool is_common_dpas = __ESIMD_DNS::is_fp_or_dword_type<T>::value;

  static_assert((is_4xhf || is_4xbf || is_common_dpas),
                "unsupported dpas type");

  static_assert(__ESIMD_DNS::is_dword_type<T1>::value,
                "Src1 must be DWORD type");

  static_assert(__ESIMD_DNS::is_dword_type<T2>::value,
                "Src2 must be DWORD type");

  static_assert((N == 8 * repeat_count) || (N == 16 * repeat_count),
                "Execution size must be 8 or 16");

  static_assert((systolic_depth == 8) || (systolic_depth == 4),
                "systolic_depth must be 8 or 4");

  static_assert((repeat_count >= 1) && (repeat_count <= 8),
                "repeat_count must be within 1 to 8");

  constexpr auto en_ops_per_channel =
      detail::get_ops_per_channel(src1_precision, src2_precision);
  static_assert(en_ops_per_channel != detail::dpas_ops_per_channel::INVALID,
                "invalid combination of Src1/Src2 precision");
  constexpr auto ops_per_channel = static_cast<unsigned>(en_ops_per_channel);

  constexpr auto src1_precision_bits =
      detail::get_precision_bits(src1_precision);
  static_assert(
      N1 == ((src1_precision_bits * systolic_depth * ops_per_channel * N) /
             (repeat_count * sizeof(T1) * 8)),
      "invalid size for Src1");

  constexpr auto src2_precision_bits =
      detail::get_precision_bits(src2_precision);
  static_assert(N2 == ((src2_precision_bits * systolic_depth * ops_per_channel *
                        ((repeat_count + 1) / 2)) /
                       (sizeof(T2) * 8)),
                "invalid size for Src2");

#if defined(__SYCL_DEVICE_ONLY__)
  int dpas_info = (repeat_count << 24) + (systolic_depth << 16) +
                  (((int)src2_precision + 1) << 8) + ((int)src1_precision + 1);
  __ESIMD_NS::simd<T, N> result = __esimd_dpasw<T, T1, T2, N, N1, N2>(
      src0.data(), src1.data(), src2.data(), dpas_info);
#else
  __ESIMD_NS::simd<T, N> result =
      __esimd_dpasw<src1_precision, src2_precision, systolic_depth,
                    repeat_count, T, T1, T2, N, N1, N2>(
          src0.data(), src1.data(), src2.data());
#endif // __SYCL_DEVICE_ONLY__

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return result;
  else
    return __ESIMD_NS::saturate<T>(result);
}

/// DPASW2
/// @param src1 is the first source perand with data precision type specified
/// by src1_precision.
/// @param src2 is the second source operand with data precision type specified
/// by src2_precision.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return the vector value of DPAS computation result.
template <argument_type src1_precision, argument_type src2_precision,
          int systolic_depth, int repeat_count, typename T, typename T1,
          typename T2, int N, int N1, int N2,
          typename Sat = __ESIMD_NS::saturation_off_tag>
__ESIMD_API __ESIMD_NS::simd<T, N> dpasw2(__ESIMD_NS::simd<T1, N1> src1,
                                          __ESIMD_NS::simd<T2, N2> src2,
                                          Sat sat = {}) {
  constexpr bool is_4xhf =
      (__ESIMD_DNS::is_type<T, cl::sycl::detail::half_impl::StorageT>()) &&
      src1_precision == src2_precision && src1_precision == argument_type::FP16;

  constexpr bool is_4xbf = __ESIMD_DNS::is_word_type<T>::value &&
                           src1_precision == src2_precision &&
                           src1_precision == argument_type::BF16;

  constexpr bool is_common_dpas = __ESIMD_DNS::is_fp_or_dword_type<T>::value;

  static_assert((is_4xhf || is_4xbf || is_common_dpas),
                "unsupported dpas type");

  static_assert(__ESIMD_DNS::is_dword_type<T1>::value,
                "Src1 must be DWORD type");

  static_assert(__ESIMD_DNS::is_dword_type<T2>::value,
                "Src2 must be DWORD type");

  static_assert((N == 8 * repeat_count) || (N == 16 * repeat_count),
                "Execution size must be 8 or 16");

  static_assert((systolic_depth == 8) || (systolic_depth == 4),
                "systolic_depth must be 8 or 4");

  static_assert((repeat_count >= 1) && (repeat_count <= 8),
                "repeat_count must be within 1 to 8");

  constexpr auto en_ops_per_channel =
      detail::get_ops_per_channel(src1_precision, src2_precision);
  static_assert(en_ops_per_channel != detail::dpas_ops_per_channel::INVALID,
                "invalid combination of Src1/Src2 precision");
  constexpr auto ops_per_channel = static_cast<unsigned>(en_ops_per_channel);

  constexpr auto src1_precision_bits =
      detail::get_precision_bits(src1_precision);
  static_assert(
      N1 == ((src1_precision_bits * systolic_depth * ops_per_channel * N) /
             (repeat_count * sizeof(T1) * 8)),
      "invalid size for Src1");

  constexpr auto src2_precision_bits =
      detail::get_precision_bits(src2_precision);
  static_assert(N2 == ((src2_precision_bits * systolic_depth * ops_per_channel *
                        ((repeat_count + 1) / 2)) /
                       (sizeof(T2) * 8)),
                "invalid size for Src2");

#if defined(__SYCL_DEVICE_ONLY__)
  int dpas_info = (repeat_count << 24) + (systolic_depth << 16) +
                  (((int)src2_precision + 1) << 8) + ((int)src1_precision + 1);
  __ESIMD_NS::simd<T, N> result =
      __esimd_dpasw2<T, T1, T2, N, N1, N2>(src1.data(), src2.data(), dpas_info);
#else
  __ESIMD_NS::simd<T, N> result =
      __esimd_dpasw2<src1_precision, src2_precision, systolic_depth,
                     repeat_count, T, T1, T2, N, N1, N2>(src1.data(),
                                                         src2.data());
#endif // __SYCL_DEVICE_ONLY__

  if constexpr (std::is_same_v<Sat, __ESIMD_NS::saturation_off_tag>)
    return result;
  else
    return __ESIMD_NS::saturate<T>(result);
}
/// @} sycl_esimd_systolic_array_api

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
