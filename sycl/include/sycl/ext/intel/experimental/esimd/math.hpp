//==-------------- math.hpp - DPC++ Explicit SIMD API   --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement Explicit SIMD math APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/host_util.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/math_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/operators.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>
#include <sycl/ext/intel/experimental/esimd/simd.hpp>
#include <sycl/ext/intel/experimental/esimd/simd_view.hpp>

#include <cstdint>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

/// Conversion of input vector elements of type \p T1 into vector of elements of
/// type \p T0 with saturation.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector.
/// \tparam SZ size of the input and returned vector.
/// @param src the input vector.
/// @return vector of elements converted to \p T0 with saturation.
template <typename T0, typename T1, int SZ>
__ESIMD_API simd<T0, SZ> saturate(simd<T1, SZ> src) {
  if constexpr (std::is_floating_point<T0>::value)
    return __esimd_sat<T0, T1, SZ>(src.data());
  else if constexpr (std::is_floating_point<T1>::value) {
    if constexpr (std::is_unsigned<T0>::value)
      return __esimd_fptoui_sat<T0, T1, SZ>(src.data());
    else
      return __esimd_fptosi_sat<T0, T1, SZ>(src.data());
  } else if constexpr (std::is_unsigned<T0>::value) {
    if constexpr (std::is_unsigned<T1>::value)
      return __esimd_uutrunc_sat<T0, T1, SZ>(src.data());
    else
      return __esimd_ustrunc_sat<T0, T1, SZ>(src.data());
  } else {
    if constexpr (std::is_signed<T1>::value)
      return __esimd_sutrunc_sat<T0, T1, SZ>(src.data());
    else
      return __esimd_sstrunc_sat<T0, T1, SZ>(src.data());
  }
}

// abs
namespace detail {

template <typename T0, typename T1, int SZ>
ESIMD_NODEBUG ESIMD_INLINE simd<T0, SZ>
__esimd_abs_common_internal(simd<T1, SZ> src0, int flag = saturation_off) {
  simd<T1, SZ> Result = simd<T0, SZ>(__esimd_abs<T1, SZ>(src0.data()));
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T0>(Result);
}

template <typename T0, typename T1>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<T0>::value &&
                                      detail::is_esimd_scalar<T1>::value,
                                  std::remove_const_t<T0>>
    __esimd_abs_common_internal(T1 src0, int flag = saturation_off) {
  using TT0 = std::remove_const_t<T0>;
  using TT1 = std::remove_const_t<T1>;

  simd<TT1, 1> Src0 = src0;
  simd<TT0, 1> Result = __esimd_abs_common_internal<TT0>(Src0, flag);
  return Result[0];
}
} // namespace detail

/// Get absolute value (vector version)
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector.
/// \tparam SZ size of the input and returned vector.
/// @param src0 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of absolute values.
template <typename T0, typename T1, int SZ>
__ESIMD_API std::enable_if_t<
    !std::is_same<std::remove_const_t<T0>, std::remove_const_t<T1>>::value,
    simd<T0, SZ>>
abs(simd<T1, SZ> src0, int flag = saturation_off) {
  return detail::__esimd_abs_common_internal<T0, T1, SZ>(src0.data(), flag);
}

/// Get absolute value (scalar version)
/// \tparam T0 element type of the returned value.
/// \tparam T1 element type of the input value.
/// @param src0 the source operand.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return absolute value.
template <typename T0, typename T1>
__ESIMD_API std::enable_if_t<
    !std::is_same<std::remove_const_t<T0>, std::remove_const_t<T1>>::value &&
        detail::is_esimd_scalar<T0>::value &&
        detail::is_esimd_scalar<T1>::value,
    std::remove_const_t<T0>>
abs(T1 src0, int flag = saturation_off) {
  return detail::__esimd_abs_common_internal<T0, T1>(src0, flag);
}

/// Get absolute value (vector version). This is a specialization of a version
/// with three template parameters, where the element types of the input and
/// output vector are the same.
/// \tparam T1 element type of the input and output vectors.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of absolute values.
template <typename T1, int SZ>
__ESIMD_API simd<T1, SZ> abs(simd<T1, SZ> src0, int flag = saturation_off) {
  return detail::__esimd_abs_common_internal<T1, T1, SZ>(src0.data(), flag);
}

/// Get absolute value (scalar version). This is a specialization of a version
/// with two template parameters, where the types of the input and output value
/// are the same.
/// \tparam T1 element type of the input and output value.
/// @param src0 the source operand.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return absolute value.
template <typename T1>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T1>::value,
                             std::remove_const_t<T1>>
abs(T1 src0, int flag = saturation_off) {
  return detail::__esimd_abs_common_internal<T1, T1>(src0, flag);
}

/// Shift left operation (vector version)
/// \tparam T0 element type of the returned vector. Must be any integer type.
/// \tparam T1 element type of the input vector. Must be any integer type.
/// \tparam SZ size of the input and returned vector.
/// \tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted left values.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             simd<T0, SZ>>
shl(simd<T1, SZ> src0, U src1, int flag = saturation_off) {
  using ComputationTy = detail::computation_type_t<decltype(src0), U>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;

  if (flag != saturation_on) {
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
/// \tparam T0 element type of the returned value. Must be any integer type.
/// \tparam T1 element type of the input value. Must be any integer type.
/// \tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted left value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<
    detail::is_esimd_scalar<T0>::value && detail::is_esimd_scalar<T1>::value &&
        detail::is_esimd_scalar<T2>::value && std::is_integral<T0>::value &&
        std::is_integral<T1>::value && std::is_integral<T2>::value,
    std::remove_const_t<T0>>
shl(T1 src0, T2 src1, int flag = saturation_off) {
  using ComputationTy = detail::computation_type_t<T1, T2>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  simd<T0, 1> Result = esimd::shl<T0>(Src0, Src1, flag);
  return Result[0];
}

/// Shift right operation (vector version)
/// \tparam T0 element type of the returned vector. Must be any integer type.
/// \tparam T1 element type of the input vector. Must be any integer type.
/// \tparam SZ size of the input and returned vector.
/// \tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted right values.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             simd<T0, SZ>>
shr(simd<T1, SZ> src0, U src1, int flag = saturation_off) {
  using ComputationTy = detail::computation_type_t<decltype(src0), U>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  typename detail::simd_type<ComputationTy>::type Result =
      Src0.data() >> Src1.data();

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T0>(Result);
}

/// Shift right operation (scalar version)
/// \tparam T0 element type of the returned value. Must be any integer type.
/// \tparam T1 element type of the input value. Must be any integer type.
/// \tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted right value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<
    detail::is_esimd_scalar<T0>::value && detail::is_esimd_scalar<T1>::value &&
        detail::is_esimd_scalar<T2>::value && std::is_integral<T0>::value &&
        std::is_integral<T1>::value && std::is_integral<T2>::value,
    std::remove_const_t<T0>>
shr(T1 src0, T2 src1, int flag = saturation_off) {
  using ComputationTy = detail::computation_type_t<T1, T2>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  simd<T0, 1> Result = esimd::shr<T0>(Src0, Src1, flag);
  return Result[0];
}

/// Rotate left operation with two vector inputs
/// \tparam T0 element type of the returned vector. Must be any integer type.
/// \tparam T1 element type of the input vector. Must be any integer type.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the vector with number of bit positions by which the elements of
/// the input vector \p src0 shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ>
__ESIMD_API
    std::enable_if_t<std::is_integral<T0>::value && std::is_integral<T1>::value,
                     simd<T0, SZ>>
    rol(simd<T1, SZ> src0, simd<T1, SZ> src1) {
  return __esimd_rol<T0, T1, SZ>(src0.data(), src1.data());
}

/// Rotate left operation with a vector and a scalar inputs
/// \tparam T0 element type of the returned vector. Must be any integer type.
/// \tparam T1 element type of the input vector. Must be any integer type.
/// \tparam SZ size of the input and returned vectors.
/// \tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             simd<T0, SZ>>
rol(simd<T1, SZ> src0, U src1) {
  using ComputationTy = detail::computation_type_t<decltype(src0), U>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  return __esimd_rol<T0>(Src0.data(), Src1.data());
}

/// Rotate left operation with two scalar inputs
/// \tparam T0 element type of the returned value. Must be any integer type.
/// \tparam T1 element type of the input value. Must be any integer type.
/// \tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return rotated left value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<
    detail::is_esimd_scalar<T0>::value && detail::is_esimd_scalar<T1>::value &&
        detail::is_esimd_scalar<T2>::value && std::is_integral<T0>::value &&
        std::is_integral<T1>::value && std::is_integral<T2>::value,
    std::remove_const_t<T0>>
rol(T1 src0, T2 src1) {
  using ComputationTy = detail::computation_type_t<T1, T2>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  simd<T0, 1> Result = esimd::rol<T0>(Src0, Src1);
  return Result[0];
}

/// Rotate right operation with two vector inputs
/// \tparam T0 element type of the returned vector. Must be any integer type.
/// \tparam T1 element type of the input vector. Must be any integer type.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the vector with number of bit positions by which the elements of
/// the input vector \p src0 shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ>
__ESIMD_API
    std::enable_if_t<std::is_integral<T0>::value && std::is_integral<T1>::value,
                     simd<T0, SZ>>
    ror(simd<T1, SZ> src0, simd<T1, SZ> src1) {
  return __esimd_ror<T0, T1, SZ>(src0.data(), src1.data());
}

/// Rotate right operation with a vector and a scalar inputs
/// \tparam T0 element type of the returned vector. Must be any integer type.
/// \tparam T1 element type of the input vector. Must be any integer type.
/// \tparam SZ size of the input and returned vectors.
/// \tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             simd<T0, SZ>>
ror(simd<T1, SZ> src0, U src1) {
  using ComputationTy = detail::computation_type_t<decltype(src0), U>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  return __esimd_ror<T0>(Src0.data(), Src1.data());
}

/// Rotate right operation with two scalar inputs
/// \tparam T0 element type of the returned value. Must be any integer type.
/// \tparam T1 element type of the input value. Must be any integer type.
/// \tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return rotated right value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<
    detail::is_esimd_scalar<T0>::value && detail::is_esimd_scalar<T1>::value &&
        detail::is_esimd_scalar<T2>::value && std::is_integral<T0>::value &&
        std::is_integral<T1>::value && std::is_integral<T2>::value,
    std::remove_const_t<T0>>
ror(T1 src0, T2 src1) {
  using ComputationTy = detail::computation_type_t<T1, T2>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  simd<T0, 1> Result = esimd::ror<T0>(Src0, Src1);
  return Result[0];
}

/// Logical Shift Right (vector version)
/// \tparam T0 element type of the returned vector. Must be any integer type.
/// \tparam T1 element type of the input vector. Must be any integer type.
/// \tparam SZ size of the input and returned vectors.
/// \tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             simd<T0, SZ>>
lsr(simd<T1, SZ> src0, U src1, int flag = saturation_off) {
  using IntermedTy = detail::computation_type_t<T1, T1>;
  typedef typename std::make_unsigned<IntermedTy>::type ComputationTy;
  simd<ComputationTy, SZ> Src0 = src0;
  simd<ComputationTy, SZ> Result = Src0.data() >> src1.data();

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T0>(Result);
}

/// Logical Shift Right (scalar version)
/// \tparam T0 element type of the returned value. Must be any integer type.
/// \tparam T1 element type of the input value \p src0. Must be any integer
/// type.
/// \tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<
    detail::is_esimd_scalar<T0>::value && detail::is_esimd_scalar<T1>::value &&
        detail::is_esimd_scalar<T2>::value && std::is_integral<T0>::value &&
        std::is_integral<T1>::value && std::is_integral<T2>::value,
    std::remove_const_t<T0>>
lsr(T1 src0, T2 src1, int flag = saturation_off) {
  using ComputationTy = detail::computation_type_t<T1, T2>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  simd<T0, 1> Result = esimd::lsr<T0>(Src0, Src1, flag);
  return Result[0];
}

/// Arithmetical Shift Right (vector version)
/// \tparam T0 element type of the returned vector. Must be any integer type.
/// \tparam T1 element type of the input vector. Must be any integer type.
/// \tparam SZ size of the input and returned vectors.
/// \tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API std::enable_if_t<std::is_integral<T0>::value &&
                                 std::is_integral<T1>::value &&
                                 std::is_integral<U>::value,
                             simd<T0, SZ>>
asr(simd<T1, SZ> src0, U src1, int flag = saturation_off) {
  using IntermedTy = detail::computation_type_t<T1, T1>;
  typedef typename std::make_signed<IntermedTy>::type ComputationTy;
  simd<ComputationTy, SZ> Src0 = src0;
  simd<ComputationTy, SZ> Result = Src0 >> src1;

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T0>(Result);
}

/// Arithmetical Shift Right (scalar version)
/// \tparam T0 element type of the returned value. Must be any integer type.
/// \tparam T1 element type of the input value \p src0. Must be any integer
/// type.
/// \tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted value.
template <typename T0, typename T1, typename T2>
__ESIMD_API std::enable_if_t<
    detail::is_esimd_scalar<T0>::value && detail::is_esimd_scalar<T1>::value &&
        detail::is_esimd_scalar<T2>::value && std::is_integral<T0>::value &&
        std::is_integral<T1>::value && std::is_integral<T2>::value,
    std::remove_const_t<T0>>
asr(T1 src0, T2 src1, int flag = saturation_off) {
  using ComputationTy = detail::computation_type_t<T1, T2>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
  simd<T0, 1> Result = esimd::asr<T0>(Src0, Src1, flag);
  return Result[0];
}

// imul
#ifndef ESIMD_HAS_LONG_LONG
// use mulh instruction for high half
template <typename T0, typename T1, typename U, int SZ>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_dword_type<T0>::value &&
                                      detail::is_dword_type<T1>::value &&
                                      detail::is_dword_type<U>::value,
                                  simd<T0, SZ>>
    imul(simd<T0, SZ> &rmd, simd<T1, SZ> src0, U src1) {
  using ComputationTy = detail::computation_type_t<decltype(src0), U>;
  typename detail::simd_type<ComputationTy>::type Src0 = src0;
  typename detail::simd_type<ComputationTy>::type Src1 = src1;
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
__ESIMD_API std::enable_if_t<detail::is_dword_type<T0>::value &&
                                 detail::is_dword_type<T1>::value &&
                                 detail::is_dword_type<U>::value && SZ == 1,
                             simd<T0, SZ>>
imul(simd<T0, SZ> &rmd, simd<T1, SZ> src0, U src1) {
  using ComputationTy = detail::computation_type_t<decltype(rmd), long long>;
  ComputationTy Product = convert<long long>(src0);
  Product *= src1;
  rmd = Product.bit_cast_view<T0>().select<1, 1>[0];
  return Product.bit_cast_view<T0>().select<1, 1>[1];
}

template <typename T0, typename T1, typename U, int SZ>
__ESIMD_API std::enable_if_t<detail::is_dword_type<T0>::value &&
                                 detail::is_dword_type<T1>::value &&
                                 detail::is_dword_type<U>::value && SZ != 1,
                             simd<T0, SZ>>
imul(simd<T0, SZ> &rmd, simd<T1, SZ> src0, U src1) {
  using ComputationTy = detail::computation_type_t<decltype(rmd), long long>;
  ComputationTy Product = convert<long long>(src0);
  Product *= src1;
  rmd = Product.bit_cast_view<T0>().select<SZ, 2>(0);
  return Product.bit_cast_view<T0>().select<SZ, 2>(1);
}
#endif

// TODO: document
template <typename T0, typename T1, typename U, int SZ>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<U>::value, simd<T0, SZ>>
imul(simd<T0, SZ> &rmd, U src0, simd<T1, SZ> src1) {
  return esimd::imul(rmd, src1, src0);
}

// TODO: document
template <typename T0, typename T, typename U>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<T>::value &&
                                      detail::is_esimd_scalar<U>::value &&
                                      detail::is_esimd_scalar<T0>::value,
                                  T0>
    imul(simd<T0, 1> &rmd, T src0, U src1) {
  simd<T, 1> src_0 = src0;
  simd<U, 1> src_1 = src1;
  simd<T0, 1> res = esimd::imul(rmd, src_0.select_all(), src_1.select_all());
  return res[0];
}

/// Integral quotient (vector version)
/// \tparam T element type of the input and return vectors.
/// \tparam SZ size of the input and returned vectors.
/// \tparam U type of scalar operand \p src1.
/// @param src0 the dividend input vector.
/// @param src1 the divisor scalar value.
/// @return vector of quotient elements.
template <typename T, int SZ, typename U>
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                     simd<T, SZ>>
    quot(simd<T, SZ> src0, U src1) {
  return src0 / src1;
}

/// Integral quotient (scalar version)
/// \tparam T0 element type of the dividend \p src0 and returned value.
/// \tparam T1 element type of the divisor \p src1.
/// @param src0 the dividend.
/// @param src1 the divisor.
/// @return quotient value.
template <typename T0, typename T1>
__ESIMD_API std::enable_if_t<
    detail::is_esimd_scalar<T0>::value && detail::is_esimd_scalar<T1>::value &&
        std::is_integral<T0>::value && std::is_integral<T1>::value,
    std::remove_const_t<T0>>
quot(T0 src0, T1 src1) {
  return src0 / src1;
}

/// Modulo (vector version)
/// \tparam T element type of the input and return vectors.
/// \tparam SZ size of the input and returned vectors.
/// \tparam U type of scalar operand \p src1.
/// @param src0 the dividend input vector.
/// @param src1 the divisor scalar value.
/// @return vector of elements after applying modulo operation.
template <typename T, int SZ, typename U>
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                     simd<T, SZ>>
    mod(simd<T, SZ> src0, U src1) {
  return src0 % src1;
}

/// Modulo (scalar version)
/// \tparam T0 element type of the dividend \p src0 and returned value.
/// \tparam T1 element type of the divisor \p src1.
/// @param src0 the dividend.
/// @param src1 the divisor.
/// @return Modulo value.
template <typename T0, typename T1>
__ESIMD_API std::enable_if_t<
    detail::is_esimd_scalar<T0>::value && detail::is_esimd_scalar<T1>::value &&
        std::is_integral<T0>::value && std::is_integral<T1>::value,
    std::remove_const_t<T0>>
mod(T0 src0, T1 src1) {
  return src0 % src1;
}

/// Integral division with a vector dividend and a scalar divisor. Computes
/// quotient and remainder of division. \tparam T element type of the input and
/// return vectors. \tparam SZ size of the input and returned vectors. \tparam U
/// type of scalar operand \p src1.
/// @param[out] remainder the vector of remainders from a division operation.
/// @param src0 the dividend input vector.
/// @param src1 the divisor scalar value.
/// @return vector of quotient elements.
template <typename T, int SZ, typename U>
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value,
                     simd<T, SZ>>
    div(simd<T, SZ> &remainder, simd<T, SZ> src0, U src1) {
  remainder = src0 % src1;
  return src0 / src1;
}

/// Integral division with a scalar dividend and a vector divisor. Computes
/// quotient and remainder of division. \tparam T element type of the input and
/// return vectors. \tparam SZ size of the input and returned vectors. \tparam U
/// type of scalar operand \p src1.
/// @param[out] remainder the vector of remainders from a division operation.
/// @param src0 the dividend scalar value.
/// @param src1 the divisor input vector.
/// @return vector of quotient elements.
template <typename T, int SZ, typename U>
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && std::is_integral<U>::value &&
                         detail::is_esimd_scalar<U>::value,
                     simd<T, SZ>>
    div(simd<T, SZ> &remainder, U src0, simd<T, SZ> src1) {
  remainder = src0 % src1;
  return src0 / src1;
}

/// Integral division (scalar version). Computes quotient and remainder of
/// division.
/// \tparam RT element type of the output remainder vector.
/// \tparam T0 element type of the dividend \p src0.
/// \tparam T1 element type of the divisor \p src1.
/// @param[out] remainder the vector of size 1 with a remainder from division.
/// @param src0 the dividend scalar value.
/// @param src1 the divisor scalar value.
/// @return scalar quotient value.
template <typename RT, typename T0, typename T1>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<RT>::value &&
                                      detail::is_esimd_scalar<T0>::value &&
                                      detail::is_esimd_scalar<T1>::value,
                                  std::remove_const_t<RT>>
    div(simd<std::remove_const_t<RT>, 1> &remainder, T0 src0, T1 src1) {
  remainder[0] = src0 % src1;
  return src0 / src1;
}

/// Selects component-wise the maximum of the two vectors.
/// The source operands must be both of integer or both of floating-point type.
/// \tparam T element type of the input and return vectors.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.
template <typename T, int SZ>
__ESIMD_API simd<T, SZ> max(simd<T, SZ> src0, simd<T, SZ> src1,
                            int flag = saturation_off) {
  if constexpr (std::is_floating_point<T>::value) {
    auto Result = __esimd_fmax<T, SZ>(src0.data(), src1.data());
    Result = (flag == saturation_off) ? Result : __esimd_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  } else if constexpr (std::is_unsigned<T>::value) {
    auto Result = __esimd_umax<T, SZ>(src0.data(), src1.data());
    Result = (flag == saturation_off) ? Result
                                      : __esimd_uutrunc_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  } else {
    auto Result = __esimd_smax<T, SZ>(src0.data(), src1.data());
    Result = (flag == saturation_off) ? Result
                                      : __esimd_sstrunc_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  }
}

/// Selects maximums for each element of the input vector and a scalar.
/// The source operands must be both of integer or both of
/// floating-point type.
/// \tparam T element type of the input and return vectors.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.
template <typename T, int SZ>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T>::value, simd<T, SZ>>
max(simd<T, SZ> src0, T src1, int flag = saturation_off) {
  simd<T, SZ> Src1 = src1;
  simd<T, SZ> Result = esimd::max<T>(src0, Src1, flag);
  return Result;
}

/// Selects maximums for each element of the input scalar and a vector.
/// The source operands must be both of integer or both of
/// floating-point type.
/// \tparam T element type of the input and return vectors.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the scalar value.
/// @param src1 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.
template <typename T, int SZ>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T>::value, simd<T, SZ>>
max(T src0, simd<T, SZ> src1, int flag = saturation_off) {
  simd<T, SZ> Src0 = src0;
  simd<T, SZ> Result = esimd::max<T>(Src0, src1, flag);
  return Result;
}

/// Selects maximum between two scalar values. (scalar version)
/// The source operands must be both of integer or both of floating-point type.
/// \tparam T element type of the input and return vectors.
/// @param src0 the scalar value.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return maximum value between the two inputs.
template <typename T>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<T>::value, T>
    max(T src0, T src1, int flag = saturation_off) {
  simd<T, 1> Src0 = src0;
  simd<T, 1> Src1 = src1;
  simd<T, 1> Result = esimd::max<T>(Src0, Src1, flag);
  return Result[0];
}

/// Selects component-wise the minimum of the two vectors.
/// The source operands must be both of integer or both of floating-point type.
/// \tparam T element type of the input and return vectors.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.
template <typename T, int SZ>
__ESIMD_API simd<T, SZ> min(simd<T, SZ> src0, simd<T, SZ> src1,
                            int flag = saturation_off) {
  if constexpr (std::is_floating_point<T>::value) {
    auto Result = __esimd_fmin<T, SZ>(src0.data(), src1.data());
    Result = (flag == saturation_off) ? Result : __esimd_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  } else if constexpr (std::is_unsigned<T>::value) {
    auto Result = __esimd_umin<T, SZ>(src0.data(), src1.data());
    Result = (flag == saturation_off) ? Result
                                      : __esimd_uutrunc_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  } else {
    auto Result = __esimd_smin<T, SZ>(src0.data(), src1.data());
    Result = (flag == saturation_off) ? Result
                                      : __esimd_sstrunc_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  }
}

/// Selects minimums for each element of the input vector and a scalar.
/// The source operands must be both of integer or both of
/// floating-point type.
/// \tparam T element type of the input and return vectors.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.
template <typename T, int SZ>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T>::value, simd<T, SZ>>
min(simd<T, SZ> src0, T src1, int flag = saturation_off) {
  simd<T, SZ> Src1 = src1;
  simd<T, SZ> Result = esimd::min<T>(src0, Src1, flag);
  return Result;
}

/// Selects minimums for each element of the input scalar and a vector.
/// The source operands must be both of integer or both of
/// floating-point type.
/// \tparam T element type of the input and return vectors.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the scalar value.
/// @param src1 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.
template <typename T, int SZ>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T>::value, simd<T, SZ>>
min(T src0, simd<T, SZ> src1, int flag = saturation_off) {
  simd<T, SZ> Src0 = src0;
  simd<T, SZ> Result = esimd::min<T>(Src0, src1, flag);
  return Result;
}

/// Selects minimum between two scalar values.
/// The source operands must be both of integer or both of floating-point type.
/// \tparam T element type of the input and return vectors.
/// @param src0 the scalar value.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return minimum value between the two inputs.
template <typename T>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<T>::value, T>
    min(T src0, T src1, int flag = saturation_off) {
  simd<T, 1> Src0 = src0;
  simd<T, 1> Src1 = src1;
  simd<T, 1> Result = esimd::min<T>(Src0, Src1, flag);
  return Result[0];
}

// Dot product builtins
#if defined(ESIMD_GEN7_5) || defined(ESIMD_GEN8) || defined(ESIMD_GEN8_5) ||   \
    defined(ESIMD_GEN9) || defined(ESIMD_GEN9_5)

// FIXME: describe the operation better
/// Dot product on groups of 4 elements.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector.
/// \tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// \tparam U type of scalar operand \p src1.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API simd<T0, SZ> dp2(simd<T1, SZ> src0, U src1,
                             int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");
  simd<float, SZ> Src0 = src0;
  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Result = __esimd_dp2(Src0.data(), Src1.data());
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T0>(Result);
}

// FIXME: describe the operation better
/// Dot product on groups of 4 elements.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector.
/// \tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// \tparam U type of scalar operand \p src1.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API simd<T0, SZ> dp3(simd<T1, SZ> src0, U src1,
                             int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");
  simd<float, SZ> Src0 = src0;
  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Result = __esimd_dp3(Src0.data(), Src1.data());
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T0>(Result);
}

// FIXME: describe the operation better
/// Dot product on groups of 4 elements.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector.
/// \tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// \tparam U type of scalar operand \p src1.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U>
__ESIMD_API simd<T0, SZ> dp4(simd<T1, SZ> src0, U src1,
                             int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");
  simd<float, SZ> Src0 = src0;
  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Result = __esimd_dp4(Src0.data(), Src1.data());
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T0>(Result);
}

// FIXME: describe the operation better
/// Dot product on groups of 4 elements.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector.
/// \tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// \tparam U type of scalar operand \p src1.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, typename U, int SZ>
__ESIMD_API simd<T0, SZ> dph(simd<T1, SZ> src0, U src1,
                             int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");
  simd<float, SZ> Src0 = src0;
  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Result = __esimd_dph(Src0.data(), Src1.data());
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T0>(Result);
}

// FIXME: describe the operation better
/// Linear equation.
/// \tparam RT element type of the output vector.
/// \tparam T1 element type of the first input vector \p src0.
/// \tparam T2 element type of the second input vector \p src1.
/// \tparam SZ size of the second input vector and returned vectors. Must be a
/// multiple of 4.
/// @param src0 the first input vector of size 4.
/// @param src1 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return resulting vector from linear equation operation.
template <typename RT, typename T1, typename T2, int SZ>
__ESIMD_API simd<RT, SZ> line(simd<T1, 4> src0, simd<T2, SZ> src1,
                              int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  simd<float, 4> Src0 = src0;
  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Result = __esimd_line(Src0.data(), Src1.data());

  simd<RT, SZ> Result;
  if (flag == saturation_on)
    Result = esimd::saturate<RT>(Result);
  else
    Result = Result;

  return Result;
}

/// FIXME: linear equation.
/// \tparam RT element type of the output vector.
/// \tparam T element type of the first input vector \p src0.
/// \tparam SZ size of the second input vector and returned vectors. Must be a
/// multiple of 4.
/// @param P the first input value.
/// @param Q the second input value.
/// @param src1 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return resulting vector from linear equation operation.
template <typename RT, typename T, int SZ>
__ESIMD_API simd<RT, SZ> line(float P, float Q, simd<T, SZ> src1,
                              int flag = saturation_off) {
  simd<float, 4> Src0 = P;
  Src0(3) = Q;
  return esimd::line<RT>(Src0, src1, flag);
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

/// FIXME: Dot product on groups of 4 elements.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector. Must be a float type.
/// \tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// \tparam U type of scalar operand \p src1. Must be a float type.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_fp_or_dword_type<T1>::value &&
                                      std::is_floating_point<T1>::value &&
                                      detail::is_fp_or_dword_type<U>::value &&
                                      std::is_floating_point<U>::value,
                                  simd<T0, SZ>>
    dp2(simd<T1, SZ> src0, U src1, int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[i] * Src1[i] + src0[i + 1] * Src1[i + 1];
  }
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T1>(Result);
}

/// FIXME: Dot product on groups of 4 elements.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector. Must be a float type.
/// \tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// \tparam U type of scalar operand \p src1. Must be a float type.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_fp_or_dword_type<T1>::value &&
                                      std::is_floating_point<T1>::value &&
                                      detail::is_fp_or_dword_type<U>::value &&
                                      std::is_floating_point<U>::value,
                                  simd<T0, SZ>>
    dp3(simd<T1, SZ> src0, U src1, int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[i] * Src1[i] + src0[i + 1] * Src1[i + 1] +
                             src0[i + 2] * Src1[i + 2];
  }
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T1>(Result);
}

/// FIXME: Dot product on groups of 4 elements.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector. Must be a float type.
/// \tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// \tparam U type of scalar operand \p src1. Must be a float type.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T0, typename T1, int SZ, typename U>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_fp_or_dword_type<T1>::value &&
                                      std::is_floating_point<T1>::value &&
                                      detail::is_fp_or_dword_type<U>::value &&
                                      std::is_floating_point<U>::value,
                                  simd<T0, SZ>>
    dp4(simd<T1, SZ> src0, U src1, int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  simd<T1, SZ> Src1 = src1;
  simd<float, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[i] * Src1[i] + src0[i + 1] * Src1[i + 1] +
                             src0[i + 2] * Src1[i + 2] +
                             src0[i + 3] * Src1[i + 3];
  }
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T1>(Result);
}

/// FIXME: Dot product on groups of 4 elements.
/// \tparam T0 element type of the returned vector.
/// \tparam T1 element type of the input vector. Must be a float type.
/// \tparam SZ size of the input and returned vectors. Must be a multiple of 4.
/// \tparam U type of scalar operand \p src1. Must be a float type.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of elements.
template <typename T, typename U, int SZ>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_fp_or_dword_type<T>::value &&
                                      std::is_floating_point<T>::value &&
                                      detail::is_fp_or_dword_type<U>::value &&
                                      std::is_floating_point<U>::value,
                                  simd<T, SZ>>
    dph(simd<T, SZ> src0, U src1, int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[i] * Src1[i] + src0[i + 1] * Src1[i + 1] +
                             src0[i + 2] * Src1[i + 2] + 1.0 * Src1[i + 3];
  }
  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T>(Result);
}

/// FIXME: linear equation.
/// \tparam T element type of the second input vector \p src1 and returned
/// vector. Must be a float type.
/// \tparam SZ size of the second input vector and returned vectors.
/// Must be a multiple of 4.
/// @param src0 the first input vector of size 4.
/// @param src1 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return resulting vector from linear equation operation.
template <typename T, int SZ>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_fp_or_dword_type<T>::value &&
                                      std::is_floating_point<T>::value,
                                  simd<T, SZ>>
    line(simd<T, 4> src0, simd<T, SZ> src1, int flag = saturation_off) {
  static_assert(SZ % 4 == 0, "result size is not a multiple of 4");

  simd<T, SZ> Src1 = src1;
  simd<T, SZ> Result;
#pragma unroll
  for (int i = 0; i < SZ; i += 4) {
    Result.select<4, 1>(i) = src0[0] * src1[i] + src0[3];
  }

  if (flag == saturation_on)
    Result = esimd::saturate<T>(Result);

  return Result;
}

/// FIXME: linear equation.
/// \tparam T element type of the first input vector \p src0. Must be a float
/// type.
/// \tparam SZ size of the second input vector and returned vectors. Must
/// be a multiple of 4.
/// @param P the first input value.
/// @param Q the second input value.
/// @param src1 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return resulting vector from linear equation operation.
template <typename T, int SZ>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_fp_or_dword_type<T>::value &&
                                      std::is_floating_point<T>::value,
                                  simd<T, SZ>>
    line(float P, float Q, simd<T, SZ> src1, int flag = saturation_off) {
  simd<T, 4> Src0 = P;
  Src0(3) = Q;
  return esimd::line<T>(Src0, src1, flag);
}

#endif

/// Performs component-wise truncate-to-minus-infinity fraction operation of
/// \p src0. (vector version)
/// \tparam T element type of the input vector \p src0 and returned vector.
/// \tparam SZ size of the second input vector and returned vectors.
/// @param src0 the input vector.
/// @return vector of elements after fraction operation.
template <typename T, int SZ> __ESIMD_API simd<T, SZ> frc(simd<T, SZ> src0) {
  simd<float, SZ> Src0 = src0;
  return __esimd_frc(Src0.data());
}

/// Performs truncate-to-minus-infinity fraction operation of \p src0.
/// (scalar version)
/// \tparam T element type of the input \p src0 and returned value.
/// @param src0 the input scalar value.
/// @return result of a fraction operation.
template <typename T> __ESIMD_API T frc(T src0) {
  simd<T, 1> Src0 = src0;
  simd<T, 1> Result = esimd::frc<T>(Src0);
  return Result[0];
}

// lzd
template <typename RT, typename T0, int SZ>
__ESIMD_API simd<RT, SZ> lzd(simd<T0, SZ> src0, int flag = saturation_off) {
  // Saturation parameter ignored
  simd<uint, SZ> Src0 = src0;
  return __esimd_lzd<uint>(Src0.data());
}

template <typename RT, typename T0>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<RT>::value &&
                                      detail::is_esimd_scalar<T0>::value,
                                  std::remove_const_t<RT>>
    lzd(T0 src0, int flag = saturation_off) {
  simd<T0, 1> Src0 = src0;
  simd<RT, 1> Result = esimd::lzd<RT>(Src0);
  return Result[0];
}

// lrp
#if defined(ESIMD_GEN7_5) || defined(ESIMD_GEN8) || defined(ESIMD_GEN8_5) ||   \
    defined(ESIMD_GEN9) || defined(ESIMD_GEN9_5)

template <int SZ, typename U, typename V>
__ESIMD_API simd<float, SZ> lrp(simd<float, SZ> src0, U src1, V src2,
                                int flag = saturation_off) {
  static_assert(SZ >= 4 && (SZ & 0x3) == 0,
                "vector size must be a multiple of 4");
  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Src2 = src2;
  simd<float, SZ> Result =
      __esimd_lrp<SZ>(src0.data(), Src1.data(), Src2.data());

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<float>(Result);
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
template <typename T, int SZ, typename U, typename V>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_fp_or_dword_type<T>::value &&
                                      std::is_floating_point<T>::value &&
                                      detail::is_fp_or_dword_type<U>::value &&
                                      std::is_floating_point<U>::value,
                                  simd<T, SZ>>
    lrp(simd<T, SZ> src0, U src1, V src2, int flag = saturation_off) {

  simd<float, SZ> Src1 = src1;
  simd<float, SZ> Src2 = src2;
  simd<float, SZ> Result;
  Result = Src1 * src0 + Src2 * (1.0f - src0);
  if (flag != saturation_on)
    return Result;
  return esimd::saturate<T>(Result);
}
#endif

// pln
template <int SZ>
__ESIMD_API simd<float, SZ> pln(simd<float, 4> src0, simd<float, SZ> src1,
                                simd<float, SZ> src2,
                                int flag = saturation_off) {
  static_assert(SZ >= 8 && (SZ & 0x7) == 0,
                "vector size must be a multiple of 8");

  // __esimd_intrinsic_impl_pln() requires src1 and src2 to be combined into
  // a single matrix, interleaving the values together in blocks of 8
  // items (ie, a block of 8 from src1, then a block of 8 from src2, then
  // the next block of 8 from src1, then the next block of 8 from src2,
  // and so-on.)
  simd<float, (SZ >> 3) * 16> Src12v;
  auto Src12 = Src12v.template bit_cast_view<float, (SZ >> 3), 16>();

  Src12.select<(SZ >> 3), 1, 8, 1>(0, 0) =
      src1.template bit_cast_view<float, (SZ >> 3), 8>();
  Src12.select<(SZ >> 3), 1, 8, 1>(0, 8) =
      src2.template bit_cast_view<float, (SZ >> 3), 8>();

  simd<float, SZ> Result = __esimd_pln<SZ>(src0.data(), Src12.read().data());

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<float>(Result);
}

// bf_reverse
template <typename T0, typename T1, int SZ>
__ESIMD_API simd<T0, SZ> bf_reverse(simd<T1, SZ> src0) {
  simd<unsigned, SZ> Src0 = src0;
  return __esimd_bfrev<unsigned>(Src0.data());
}

template <typename T0, typename T1>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<T0>::value &&
                                      detail::is_esimd_scalar<T1>::value,
                                  std::remove_const_t<T0>>
    bf_reverse(T1 src0) {
  simd<T1, 1> Src0 = src0;
  simd<T0, 1> Result = esimd::bf_reverse<T0>(Src0);
  return Result[0];
}

// bf_insert
template <typename T0, typename T1, int SZ, typename U, typename V, typename W>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<std::is_integral<T1>::value, simd<T0, SZ>>
    bf_insert(U src0, V src1, W src2, simd<T1, SZ> src3) {
  typedef typename detail::dword_type<T1> DT1;
  static_assert(std::is_integral<DT1>::value && sizeof(DT1) == sizeof(int),
                "operand conversion failed");
  simd<DT1, SZ> Src0 = src0;
  simd<DT1, SZ> Src1 = src1;
  simd<DT1, SZ> Src2 = src2;
  simd<DT1, SZ> Src3 = src3;

  return __esimd_bfi<DT1>(Src0.data(), Src1.data(), Src2.data(), Src3.data());
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<T0>::value &&
                                      detail::is_esimd_scalar<T4>::value,
                                  std::remove_const_t<T0>>
    bf_insert(T1 src0, T2 src1, T3 src2, T4 src3) {
  simd<T4, 1> Src3 = src3;
  simd<T0, 1> Result = esimd::bf_insert<T0>(src0, src1, src2, Src3);
  return Result[0];
}

// bf_extract
template <typename T0, typename T1, int SZ, typename U, typename V>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<std::is_integral<T1>::value, simd<T0, SZ>>
    bf_extract(U src0, V src1, simd<T1, SZ> src2) {
  typedef typename detail::dword_type<T1> DT1;
  static_assert(std::is_integral<DT1>::value && sizeof(DT1) == sizeof(int),
                "operand conversion failed");
  simd<DT1, SZ> Src0 = src0;
  simd<DT1, SZ> Src1 = src1;
  simd<DT1, SZ> Src2 = src2;

  return __esimd_sbfe<DT1>(Src0.data(), Src1.data(), Src2.data());
}

template <typename T0, typename T1, typename T2, typename T3>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<detail::is_esimd_scalar<T0>::value &&
                                      detail::is_esimd_scalar<T3>::value,
                                  std::remove_const_t<T0>>
    bf_extract(T1 src0, T2 src1, T3 src2) {
  simd<T3, 1> Src2 = src2;
  simd<T0, 1> Result = esimd::bf_extract<T0>(src0, src1, Src2);
  return Result[0];
}

////////////////////////////////////////////////////////////////////////////////
// ESIMD arithmetic intrinsics:
//
// inv, log2, exp2, sqrt, rsqrt, sin, cos
//
// share the same requirements.
//
// template <int SZ>
// simd<float, SZ>
// ESIMD_INLINE inv(simd<float, SZ> src0, int flag = saturation_off) {
//   simd<float, SZ> Result = __esimd_inv(src0);
//   if (flag != saturation_on)
//     return Result;
//   return __esimd_sat<float>(Result);
// }
//
// template <int N1, int N2>
// __ESIMD_API
// simd<float, N1 * N2>
// inv(matrix<float, N1, N2> src0, int flag = saturation_off) {
//   simd<float, N1 * N2> Src0 = src0;
//   return esimd::inv(Src0, flag);
// }
//
// ESIMD_INLINE float inv(float src0, int flag = saturation_off) {
//   simd<float, 1> Src0 = src0;
//   simd<float, 1> Result = esimd::inv(Src0, flag);
//   return Result[0];
// }
//
// we also make the scalar version template-based by adding
// a "typename T". Since the type can only be float, we hack it
// by defining T=void without instantiating it to be float.

#define ESIMD_INTRINSIC_DEF(type, name, iname)                                 \
  template <int SZ>                                                            \
  __ESIMD_API simd<type, SZ> name(simd<type, SZ> src0,                         \
                                  int flag = saturation_off) {                 \
    simd<type, SZ> Result = __esimd_##iname<SZ>(src0.data());                  \
    if (flag != saturation_on)                                                 \
      return Result;                                                           \
    return esimd::saturate<type>(Result);                                      \
  }                                                                            \
  template <typename T = void>                                                 \
  __ESIMD_API type name(type src0, int flag = saturation_off) {                \
    simd<type, 1> Src0 = src0;                                                 \
    simd<type, 1> Result = name(Src0, flag);                                   \
    return Result[0];                                                          \
  }

ESIMD_INTRINSIC_DEF(float, inv, inv)
ESIMD_INTRINSIC_DEF(float, log2, log)
ESIMD_INTRINSIC_DEF(float, exp2, exp)
ESIMD_INTRINSIC_DEF(float, sqrt, sqrt)
ESIMD_INTRINSIC_DEF(float, ieee_sqrt, sqrt_ieee)
ESIMD_INTRINSIC_DEF(float, rsqrt, rsqrt)
ESIMD_INTRINSIC_DEF(float, sin, sin)
ESIMD_INTRINSIC_DEF(float, cos, cos)

ESIMD_INTRINSIC_DEF(double, ieee_sqrt, sqrt_ieee)

#undef ESIMD_INTRINSIC_DEF

#define ESIMD_INTRINSIC_DEF(ftype, name, iname)                                \
  template <int SZ, typename U>                                                \
  __ESIMD_API simd<ftype, SZ> name(simd<ftype, SZ> src0, U src1,               \
                                   int flag = saturation_off) {                \
    simd<ftype, SZ> Src1 = src1;                                               \
    simd<ftype, SZ> Result = __esimd_##iname<SZ>(src0.data(), Src1.data());    \
    if (flag != saturation_on)                                                 \
      return Result;                                                           \
                                                                               \
    return esimd::saturate<ftype>(Result);                                     \
  }                                                                            \
  template <int SZ, typename U>                                                \
  __ESIMD_API                                                                  \
      std::enable_if_t<detail::is_esimd_scalar<U>::value, simd<ftype, SZ>>     \
      name(U src0, simd<ftype, SZ> src1, int flag = saturation_off) {          \
    simd<ftype, SZ> Src0 = src0;                                               \
    return name(Src0, src1, flag);                                             \
  }                                                                            \
  __ESIMD_API ftype name(ftype src0, ftype src1, int flag = saturation_off) {  \
    simd<ftype, 1> Src0 = src0;                                                \
    simd<ftype, 1> Src1 = src1;                                                \
    simd<ftype, 1> Result = name(Src0, Src1, flag);                            \
    return Result[0];                                                          \
  }

ESIMD_INTRINSIC_DEF(float, pow, pow)

ESIMD_INTRINSIC_DEF(float, div_ieee, ieee_div)
ESIMD_INTRINSIC_DEF(double, div_ieee, ieee_div)

#undef ESIMD_INTRINSIC_DEF

// sincos
template <int SZ, typename U>
__ESIMD_API simd<float, SZ> sincos(simd<float, SZ> &dstcos, U src0,
                                   int flag = saturation_off) {
  dstcos = esimd::cos(src0, flag);
  return esimd::sin(src0, flag);
}

// atan

#define ESIMD_HDR_CONST_PI 3.1415926535897932384626433832795

template <typename T, int SZ>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, simd<T, SZ>>
    atan(simd<T, SZ> src0, int flag = saturation_off) {
  simd<T, SZ> Src0 = esimd::abs(src0);

  simd_mask<SZ> Neg = src0 < T(0.0);
  simd_mask<SZ> Gt1 = Src0 > T(1.0);

  Src0.merge(esimd::inv(Src0), Gt1);

  simd<T, SZ> Src0P2 = Src0 * Src0;
  simd<T, SZ> Src0P4 = Src0P2 * Src0P2;

  simd<T, SZ> Result = (Src0P4 * T(0.185696) +
                        ((Src0 * T(0.787997) + T(0.63693)) * Src0P2) + Src0) /
                       (((((Src0 * -T(0.000121387) + T(0.00202308)) * Src0P2) +
                          (Src0 * -T(0.0149145)) + T(0.182569)) *
                         Src0P4) +
                        ((Src0 * T(0.395889) + T(1.12158)) * Src0P2) +
                        (Src0 * T(0.636918)) + T(1.0));

  Result.merge(Result - T(ESIMD_HDR_CONST_PI / 2.0), Gt1);
  Result.merge(Result, Neg);

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T>(Result);
}

template <typename T>
__ESIMD_API std::enable_if_t<std::is_floating_point<T>::value, T>
atan(T src0, int flag = saturation_off) {
  simd<T, 1> Src0 = src0;
  simd<T, 1> Result = esimd::atan(Src0, flag);
  return Result[0];
}

// acos

template <typename T, int SZ>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, simd<T, SZ>>
    acos(simd<T, SZ> src0, int flag = saturation_off) {
  simd<T, SZ> Src0 = esimd::abs(src0);

  simd_mask<SZ> Neg = src0 < T(0.0);
  simd_mask<SZ> TooBig = Src0 >= T(0.999998);

  // Replace oversized values to ensure no possibility of sqrt of
  // a negative value later
  Src0.merge(T(0.0), TooBig);

  simd<T, SZ> Src01m = T(1.0) - Src0;

  simd<T, SZ> Src0P2 = Src01m * Src01m;
  simd<T, SZ> Src0P4 = Src0P2 * Src0P2;

  simd<T, SZ> Result =
      (((Src01m * T(0.015098965761299077) - T(0.005516443930088506)) * Src0P4) +
       ((Src01m * T(0.047654245891495528) + T(0.163910606547823220)) * Src0P2) +
       Src01m * T(2.000291665285952400) - T(0.000007239283986332)) *
      esimd::rsqrt(Src01m * T(2.0));

  Result.merge(T(0.0), TooBig);
  Result.merge(T(ESIMD_HDR_CONST_PI) - Result, Neg);

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T>(Result);
}

template <typename T>
__ESIMD_API std::enable_if_t<std::is_floating_point<T>::value, T>
acos(T src0, int flag = saturation_off) {
  simd<T, 1> Src0 = src0;
  simd<T, 1> Result = esimd::acos(Src0, flag);
  return Result[0];
}

// asin

template <typename T, int SZ>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<std::is_floating_point<T>::value, simd<T, SZ>>
    asin(simd<T, SZ> src0, int flag = saturation_off) {
  simd_mask<SZ> Neg = src0 < T(0.0);

  simd<T, SZ> Result =
      T(ESIMD_HDR_CONST_PI / 2.0) - esimd::acos(esimd::abs(src0));

  Result.merge(-Result, Neg);

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<T>(Result);
}

template <typename T>
__ESIMD_API std::enable_if_t<std::is_floating_point<T>::value, T>
asin(T src0, int flag = saturation_off) {
  simd<T, 1> Src0 = src0;
  simd<T, 1> Result = esimd::asin(Src0, flag);
  return Result[0];
}

/// Computes the natural logarithm of the given argument. This is an
/// emulated version based on the H/W supported log2.
/// @param the source operand to compute base-e logarithm of.
/// @return the base-e logarithm of \p src0.
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE simd<float, SZ> log(simd<float, SZ> src0,
                                               int flag = saturation_off) {
  constexpr float ln2 = 0.69314718f; // std::numbers::ln2_v<float> in c++20
  simd<float, SZ> Result = esimd::log2<SZ>(src0) * ln2;

  if (flag != saturation_on)
    return Result;

  return esimd::saturate<float>(Result);
}

ESIMD_NODEBUG ESIMD_INLINE float log(float src0, int flag = saturation_off) {
  return esimd::log<1>(src0, flag)[0];
}

/// Computes e raised to the power of the given argument. This is an
/// emulated version based on the H/W supported exp2.
/// @param the source operand to compute base-e exponential of.
/// @return e raised to the power of \p src0.
template <int SZ>
ESIMD_NODEBUG ESIMD_INLINE simd<float, SZ> exp(simd<float, SZ> src0,
                                               int flag = saturation_off) {
  constexpr float log2e = 1.442695f; // std::numbers::log2e_v<float> in c++20
  return esimd::exp2<SZ>(src0 * log2e, flag);
}

ESIMD_NODEBUG ESIMD_INLINE float exp(float src0, int flag = saturation_off) {
  return esimd::exp<1>(src0, flag)[0];
}

////////////////////////////////////////////////////////////////////////////////
// Rounding intrinsics.
////////////////////////////////////////////////////////////////////////////////

#define ESIMD_INTRINSIC_DEF(name)                                              \
  template <typename T, int SZ>                                                \
  __ESIMD_API simd<T, SZ> name(simd<float, SZ> src0,                           \
                               int flag = saturation_off) {                    \
    simd<float, SZ> Result = __esimd_##name<SZ>(src0.data());                  \
    if (flag != saturation_on)                                                 \
      return Result;                                                           \
    return esimd::saturate<T>(Result);                                         \
  }                                                                            \
  template <typename T>                                                        \
  __ESIMD_API T name(float src0, int flag = saturation_off) {                  \
    simd<float, 1> Src0 = src0;                                                \
    simd<T, 1> Result = name<T>(Src0, flag);                                   \
    return Result[0];                                                          \
  }

ESIMD_INTRINSIC_DEF(rndd)
ESIMD_INTRINSIC_DEF(rndu)
ESIMD_INTRINSIC_DEF(rnde)
ESIMD_INTRINSIC_DEF(rndz)

#undef ESIMD_INTRINSIC_DEF

template <int N>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<(N == 8 || N == 16 || N == 32), uint>
    pack_mask(simd_mask<N> src0) {
  return __esimd_pack_mask<N>(src0.data());
}

template <int N>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<(N == 8 || N == 16 || N == 32), simd_mask<N>>
    unpack_mask(uint src0) {
  return __esimd_unpack_mask<N>(src0);
}

template <int N>
__ESIMD_API std::enable_if_t<(N != 8 && N != 16 && N < 32), uint>
pack_mask(simd_mask<N> src0) {
  simd_mask<(N < 8 ? 8 : N < 16 ? 16 : 32)> src_0 = 0;
  src_0.template select<N, 1>() = src0.template bit_cast_view<ushort>();
  return esimd::pack_mask(src_0);
}

/// Compare source vector elements against zero and return a bitfield combining
/// the comparison result. The representative bit in the result is set if
/// corresponding source vector element is non-zero, and is unset otherwise.
/// @param mask the source operand to be compared with zero.
/// @return an \c uint, where each bit is set if the corresponding element of
/// the source operand is non-zero and unset otherwise.
template <typename T, int N>
__ESIMD_API
    std::enable_if_t<detail::is_type<T, ushort, uint> && (N > 0 && N <= 32),
                     uint>
    ballot(simd<T, N> mask) {
  simd_mask<N> cmp = (mask != 0);
  if constexpr (N == 8 || N == 16 || N == 32) {
    return __esimd_pack_mask<N>(cmp.data());
  } else {
    constexpr int N1 = (N <= 8 ? 8 : N <= 16 ? 16 : 32);
    simd<uint16_t, N1> res = 0;
    res.template select<N, 1>() = cmp.data();
    return __esimd_pack_mask<N1>(res.data());
  }
}

/// Count number of bits set in the source operand per element.
/// @param src0 the source operand to count bits in.
/// @return a vector of \c uint32_t, where each element is set to bit count of
///     the corresponding element of the source operand.
template <typename T, int N>
ESIMD_NODEBUG
    ESIMD_INLINE std::enable_if_t<std::is_integral<T>::value && sizeof(T) <= 4,
                                  simd<uint32_t, N>>
    cbit(simd<T, N> src) {
  return __esimd_cbit<T, N>(src.data());
}

/// Scalar version of \c cbit - both input and output are scalars rather
/// than vectors.
template <typename T>
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && sizeof(T) <= 4, uint32_t>
    cbit(T src) {
  simd<T, 1> Src = src;
  simd<uint32_t, 1> Result = esimd::cbit(Src);
  return Result[0];
}

/// Scalar version of \c cbit, that takes simd_view object as an
/// argument, e.g. `cbit(v[0])`.
/// @param src0 input simd_view object of size 1.
/// @return scalar number of bits set.
template <typename BaseTy, typename RegionTy>
__ESIMD_API std::enable_if_t<
    std::is_integral<
        typename simd_view<BaseTy, RegionTy>::element_type>::value &&
        (sizeof(typename simd_view<BaseTy, RegionTy>::element_type) <= 4) &&
        (simd_view<BaseTy, RegionTy>::length == 1),
    uint32_t>
cbit(simd_view<BaseTy, RegionTy> src) {
  using Ty = typename simd_view<BaseTy, RegionTy>::element_type;
  simd<Ty, 1> Src = src;
  simd<uint32_t, 1> Result = esimd::cbit(Src);
  return Result[0];
}

/// Find the per element number of the first bit set in the source operand
/// starting from the least significant bit.
/// @param src0 the source operand to count bits in.
/// @return a vector of the same type as the source operand, where each element
///     is set to the number first bit set in corresponding element of the
///     source operand. \c 0xFFFFffff is returned for an element equal to \c 0.
/// Find component-wise the first bit from LSB side
template <typename T, int N>
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4), simd<T, N>>
    fbl(simd<T, N> src) {
  return __esimd_fbl<T, N>(src.data());
}

/// Scalar version of \c fbl - both input and output are scalars rather
/// than vectors.
template <typename T>
__ESIMD_API std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4), T>
fbl(T src) {
  simd<T, 1> Src = src;
  simd<T, 1> Result = esimd::fbl(Src);
  return Result[0];
}

/// Scalar version of \c fbl, that takes simd_view object as an
/// argument, e.g. `fbl(v[0])`.
/// @param src0 input simd_view object of size 1.
/// @return scalar number of the first bit set starting from the least
/// significant bit.
template <typename BaseTy, typename RegionTy>
__ESIMD_API std::enable_if_t<
    std::is_integral<
        typename simd_view<BaseTy, RegionTy>::element_type>::value &&
        (sizeof(typename simd_view<BaseTy, RegionTy>::element_type) == 4) &&
        (simd_view<BaseTy, RegionTy>::length == 1),
    typename simd_view<BaseTy, RegionTy>::element_type>
fbl(simd_view<BaseTy, RegionTy> src) {
  using Ty = typename simd_view<BaseTy, RegionTy>::element_type;
  simd<Ty, 1> Src = src;
  simd<Ty, 1> Result = esimd::fbl(Src);
  return Result[0];
}

/// Find the per element number of the first bit set in the source operand
/// starting from the most significant bit (sign bit is skipped).
/// @param src0 the source operand to count bits in.
/// @return a vector of the same type as the source operand, where each element
///     is set to the number first bit set in corresponding element of the
///     source operand. \c 0xFFFFffff is returned for an element equal to \c 0
///     or \c -1.
template <typename T, int N>
__ESIMD_API std::enable_if_t<std::is_integral<T>::value &&
                                 std::is_signed<T>::value && (sizeof(T) == 4),
                             simd<T, N>>
fbh(simd<T, N> src) {
  return __esimd_sfbh<T, N>(src.data());
}

/// Find the per element number of the first bit set in the source operand
/// starting from the most significant bit (sign bit is counted).
/// @param src0 the source operand to count bits in.
/// @return a vector of the same type as the source operand, where each element
///     is set to the number first bit set in corresponding element of the
///     source operand. \c 0xFFFFffff is returned for an element equal to \c 0.
template <typename T, int N>
__ESIMD_API std::enable_if_t<std::is_integral<T>::value &&
                                 !std::is_signed<T>::value && (sizeof(T) == 4),
                             simd<T, N>>
fbh(simd<T, N> src) {
  return __esimd_ufbh<T, N>(src.data());
}

/// Scalar version of \c fbh - both input and output are scalars rather
/// than vectors.
template <typename T>
__ESIMD_API std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4), T>
fbh(T src) {
  simd<T, 1> Src = src;
  simd<T, 1> Result = esimd::fbh(Src);
  return Result[0];
}

/// Scalar version of \c fbh, that takes simd_view object as an
/// argument, e.g. `fbh(v[0])`.
/// @param src0 input simd_view object of size 1.
/// @return scalar number of the first bit set starting from the most
/// significant bit.
template <typename BaseTy, typename RegionTy>
__ESIMD_API std::enable_if_t<
    std::is_integral<
        typename simd_view<BaseTy, RegionTy>::element_type>::value &&
        (sizeof(typename simd_view<BaseTy, RegionTy>::element_type) == 4) &&
        (simd_view<BaseTy, RegionTy>::length == 1),
    typename simd_view<BaseTy, RegionTy>::element_type>
fbh(simd_view<BaseTy, RegionTy> src) {
  using Ty = typename simd_view<BaseTy, RegionTy>::element_type;
  simd<Ty, 1> Src = src;
  simd<Ty, 1> Result = esimd::fbh(Src);
  return Result[0];
}

/// \brief DP4A.
///
/// @param src0 the first source operand of dp4a operation.
///
/// @param src1 the second source operand of dp4a operation.
///
/// @param src2 the third source operand of dp4a operation.
///
/// @param flag saturation flag, which has default value of saturation_off.
///
/// Returns simd vector of the dp4a operation result.
///
template <typename T1, typename T2, typename T3, typename T4, int N>
__ESIMD_API std::enable_if_t<
    detail::is_dword_type<T1>::value && detail::is_dword_type<T2>::value &&
        detail::is_dword_type<T3>::value && detail::is_dword_type<T4>::value,
    simd<T1, N>>
dp4a(simd<T2, N> src0, simd<T3, N> src1, simd<T4, N> src2,
     int flag = saturation_off) {
  simd<T2, N> Src0 = src0;
  simd<T3, N> Src1 = src1;
  simd<T4, N> Src2 = src2;
  simd<T1, N> Result;

#if defined(__SYCL_DEVICE_ONLY__)
  if (flag == saturation_off) {
    if constexpr (std::is_unsigned<T1>::value) {
      if constexpr (std::is_unsigned<T2>::value) {
        Result = __esimd_uudp4a<T1, T2, T3, T4, N>(Src0.data(), Src1.data(),
                                                   Src2.data());
      } else {
        Result = __esimd_usdp4a<T1, T2, T3, T4, N>(Src0.data(), Src1.data(),
                                                   Src2.data());
      }
    } else {
      if constexpr (std::is_unsigned<T2>::value) {
        Result = __esimd_sudp4a<T1, T2, T3, T4, N>(Src0.data(), Src1.data(),
                                                   Src2.data());
      } else {
        Result = __esimd_ssdp4a<T1, T2, T3, T4, N>(Src0.data(), Src1.data(),
                                                   Src2.data());
      }
    }
  } else {
    if constexpr (std::is_unsigned<T1>::value) {
      if constexpr (std::is_unsigned<T2>::value) {
        Result = __esimd_uudp4a_sat<T1, T2, T3, T4, N>(Src0.data(), Src1.data(),
                                                       Src2.data());
      } else {
        Result = __esimd_usdp4a_sat<T1, T2, T3, T4, N>(Src0.data(), Src1.data(),
                                                       Src2.data());
      }
    } else {
      if constexpr (std::is_unsigned<T2>::value) {
        Result = __esimd_sudp4a_sat<T1, T2, T3, T4, N>(Src0.data(), Src1.data(),
                                                       Src2.data());
      } else {
        Result = __esimd_ssdp4a_sat<T1, T2, T3, T4, N>(Src0.data(), Src1.data(),
                                                       Src2.data());
      }
    }
  }
#else
  simd<T2, N> tmp =
      __esimd_dp4a<T1, T2, T3, T4, N>(Src0.data(), Src1.data(), Src2.data());

  if (flag == saturation_on)
    Result = esimd::saturate<T1>(tmp);
  else
    Result = convert<T1>(tmp);
#endif // __SYCL_DEVICE_ONLY__

  return Result;
}

static auto constexpr ESIMD_CONST_E = 2.71828f;
static auto constexpr ESIMD_CONST_PI = 3.14159f;
static auto constexpr ESIMD_CONST_2PI = 6.28318f;

/* smallest such that 1.0+ESIMD_DBL_EPSILON != 1.0 */
static auto constexpr ESIMD_DBL_EPSILON = 0.00001f;

template <typename RT, int SZ>
ESIMD_INLINE simd<RT, SZ> floor(const simd<float, SZ> src0,
                                const uint flags = 0) {
  return esimd::rndd<RT, SZ>(src0, flags);
}

template <typename RT>
ESIMD_INLINE RT floor(const float &src0, const uint flags = 0) {
  return esimd::rndd<RT, 1U>(src0, flags)[0];
}

template <typename RT, int SZ>
ESIMD_INLINE simd<RT, SZ> ceil(const simd<float, SZ> src0,
                               const uint flags = 0) {
  return esimd::rndu<RT, SZ>(src0, flags);
}

template <typename RT>
ESIMD_INLINE RT ceil(const float &src0, const uint flags = 0) {
  return esimd::rndu<RT, 1U>(src0, flags);
}

/// Round to integral value using the round to zero rounding mode (vector
/// version).
/// \tparam RT element type of the return vector.
/// \tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of rounded values.
template <typename RT, int SZ>
__ESIMD_API simd<RT, SZ> trunc(const simd<float, SZ> &src0,
                               const uint flag = 0) {
  return esimd::rndz<RT, SZ>(src0, flag);
}

/// Round to integral value using the round to zero rounding mode (scalar
/// version).
/// \tparam RT type of the return value.
/// @param src0 the input operand.
/// @param flag enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return rounded value.
template <typename RT> __ESIMD_API RT trunc(float src0, const uint flag = 0) {
  return esimd::rndz<RT, 1U>(src0, flag)[0];
}

/* atan2_fast - a fast atan2 implementation */
/* vector input */
template <int N>
simd<float, N> atan2_fast(simd<float, N> y, simd<float, N> x,
                          const uint flags = 0);
/* scalar input */
template <typename T> float atan2_fast(T y, T x, const uint flags = 0);

/* atan2 - atan2 implementation */
/* For Vector input */
template <int N>
simd<float, N> atan2(simd<float, N> y, simd<float, N> x, const uint flags = 0);
/* scalar Input */
template <typename T> float atan2(T y, T x, const uint flags = 0);

/* fmod: */
/* vector input */
template <int N>
simd<float, N> fmod(simd<float, N> y, simd<float, N> x, const uint flags = 0);
/* scalar Input */
template <typename T> float fmod(T y, T x, const uint flags = 0);

/* sin_emu - EU emulation for sin(x) */
/* For Vector input */
template <int N> simd<float, N> sin_emu(simd<float, N> x, const uint flags = 0);
/* scalar Input */
template <typename T> float sin_emu(T x, const uint flags = 0);

/* cos_emu - EU emulation for cos(x) */
/* For Vector input */
template <int N> simd<float, N> cos_emu(simd<float, N> x, const uint flags = 0);

/* scalar Input */
template <typename T> float cos_emu(T x, const uint flags = 0);

/* tanh_cody_waite - Cody-Waite implementation for tanh(x) */
/* float input */
float tanh_cody_waite(float x);
/* vector input */
template <int N> simd<float, N> tanh_cody_waite(simd<float, N> x);
/* tanh - opencl like implementation for tanh(x) */
/* float input */
float tanh(float x);
/* vector input */
template <int N> simd<float, N> tanh(simd<float, N> x);

/* ------------------------- Extended Math Routines
 * -------------------------------------------------*/

// For vector input
template <int N>
ESIMD_INLINE simd<float, N> atan2_fast(simd<float, N> y, simd<float, N> x,
                                       const uint flags) {
  simd<float, N> a0;
  simd<float, N> a1;
  simd<float, N> atan2;

  simd_mask<N> mask = (y >= 0.0f);
  a0.merge(ESIMD_CONST_PI * 0.5f, ESIMD_CONST_PI * 1.5f, mask);
  a1.merge(0, ESIMD_CONST_PI * 2.0f, mask);

  a1.merge(ESIMD_CONST_PI, x < 0.0f);

  simd<float, N> xy = x * y;
  simd<float, N> x2 = x * x;
  simd<float, N> y2 = y * y;

  a0 -= (xy / (y2 + x2 * 0.28f + ESIMD_DBL_EPSILON));
  a1 += (xy / (x2 + y2 * 0.28f + ESIMD_DBL_EPSILON));

  atan2.merge(a1, a0, y2 <= x2);
  if (flags & saturation_on)
    atan2 = esimd::saturate<float>(atan2);
  return atan2;
}

//   For Scalar Input
template <> ESIMD_INLINE float atan2_fast(float y, float x, const uint flags) {
  simd<float, 1> vy = y;
  simd<float, 1> vx = x;
  simd<float, 1> atan2 = esimd::atan2_fast(vy, vx, flags);
  return atan2[0];
}

// atan2
// For Vector input
template <int N>
ESIMD_INLINE simd<float, N> atan2(simd<float, N> y, simd<float, N> x,
                                  const uint flags) {
  simd<float, N> v_distance;
  simd<float, N> v_y0;
  simd<float, N> atan2;
  simd_mask<N> mask;

  mask = (x < 0);
  v_y0.merge(ESIMD_CONST_PI, 0, mask);
  v_distance = esimd::sqrt(x * x + y * y);
  mask = (esimd::abs<float>(y) < 0.000001f);
  atan2.merge(v_y0, (2 * esimd::atan((v_distance - x) / y)), mask);
  if (flags & saturation_on)
    atan2 = esimd::saturate<float>(atan2);

  return atan2;
}

// For Scalar Input
template <> ESIMD_INLINE float atan2(float y, float x, const uint flags) {
  float v_distance;
  float v_y0;
  simd<float, 1> atan2;
  simd_mask<1> mask;

  mask = (x < 0);
  v_y0 = mask[0] ? ESIMD_CONST_PI : 0;
  v_distance = esimd::sqrt<float>(x * x + y * y);
  mask = (esimd::abs<float>(y) < 0.000001f);
  atan2.merge(v_y0, (2 * esimd::atan((v_distance - x) / y)), mask);
  if (flags & saturation_on)
    atan2 = esimd::saturate<float>(atan2);

  return atan2[0];
}

// fmod:
// For Vector input
template <int N>
ESIMD_INLINE simd<float, N> fmod(simd<float, N> y, simd<float, N> x,
                                 const uint flags) {
  simd<int, N> v_quot;
  simd<float, N> fmod;

  v_quot = convert<int>(y / x);
  fmod = y - x * convert<float>(v_quot);
  if (flags & saturation_on)
    fmod = esimd::saturate<float>(fmod);

  return fmod;
}

//     For Scalar Input
template <> ESIMD_INLINE float fmod(float y, float x, const uint flags) {
  int v_quot;
  simd<float, 1> fmod;

  v_quot = (int)(y / x);
  fmod = y - x * v_quot;
  if (flags & saturation_on)
    fmod = saturate<float>(fmod);

  return fmod[0];
}

static auto constexpr CMPI = 3.14159265f;

// sin_emu - EU emulation for sin(x)
// For Vector input
template <int N>
ESIMD_INLINE simd<float, N> sin_emu(simd<float, N> x, const uint flags) {
  simd<float, N> x1;
  simd<float, N> x2;
  simd<float, N> t3;

  simd<float, N> sign;
  simd<float, N> fTrig;
  simd<float, N> TwoPI(6.2831853f);
  simd<float, N> CmpI(CMPI);
  simd<float, N> OneP(1.f);
  simd<float, N> OneN(-1.f);

  x = esimd::fmod(x, TwoPI);

  x1.merge(CmpI - x, x - CmpI, (x <= CMPI));
  x1.merge(x, (x <= CMPI * 0.5f));
  x1.merge(CmpI * 2 - x, (x > CMPI * 1.5f));

  sign.merge(OneN, OneP, (x > CMPI));

  x2 = x1 * x1;
  t3 = x2 * x1 * 0.1666667f;

  fTrig =
      x1 + t3 * (OneN + x2 * 0.05f *
                            (OneP + x2 * 0.0238095f *
                                        (OneN + x2 * 0.0138889f *
                                                    (OneP - x2 * 0.0090909f))));
  fTrig *= sign;

  if (flags & saturation_on)
    fTrig = esimd::saturate<float>(fTrig);

  return fTrig;
}

// scalar Input
template <typename T> ESIMD_INLINE float sin_emu(T x0, const uint flags) {
  simd<float, 1> x1;
  simd<float, 1> x2;
  simd<float, 1> t3;

  simd<float, 1> sign;
  simd<float, 1> fTrig;
  float TwoPI = CMPI * 2.0f;

  simd<float, 1> x = esimd::fmod(x0, TwoPI);

  simd<float, 1> CmpI(CMPI);
  simd<float, 1> OneP(1.f);
  simd<float, 1> OneN(-1.f);

  x1.merge(CmpI - x, x - CmpI, (x <= CMPI));
  x1.merge(x, (x <= CMPI * 0.5f));
  x1.merge(CmpI * 2.0f - x, (x > CMPI * 1.5f));

  sign.merge(OneN, OneP, (x > CMPI));

  x2 = x1 * x1;
  t3 = x2 * x1 * 0.1666667f;

  fTrig =
      x1 + t3 * (OneN + x2 * 0.05f *
                            (OneP + x2 * 0.0238095f *
                                        (OneN + x2 * 0.0138889f *
                                                    (OneP - x2 * 0.0090909f))));
  fTrig *= sign;

  if (flags & saturation_on)
    fTrig = esimd::saturate<float>(fTrig);

  return fTrig[0];
}

// cos_emu - EU emulation for sin(x)
// For Vector input
template <int N>
ESIMD_INLINE simd<float, N> cos_emu(simd<float, N> x, const uint flags) {
  simd<float, N> x1;
  simd<float, N> x2;
  simd<float, N> t2;
  simd<float, N> t3;

  simd<float, N> sign;
  simd<float, N> fTrig;
  simd<float, N> TwoPI(6.2831853f);
  simd<float, N> CmpI(CMPI);
  simd<float, N> OneP(1.f);
  simd<float, N> OneN(-1.f);

  x = esimd::fmod(x, TwoPI);

  x1.merge(x - CMPI * 0.5f, CmpI * 1.5f - x, (x <= CMPI));
  x1.merge(CmpI * 0.5f - x, (x <= CMPI * 0.5f));
  x1.merge(x - CMPI * 1.5f, (x > CMPI * 1.5f));

  sign.merge(1, -1, ((x < CMPI * 0.5f) | (x >= CMPI * 1.5f)));

  x2 = x1 * x1;
  t3 = x2 * x1 * 0.1666667f;
  fTrig =
      x1 + t3 * (OneN + x2 * 0.05f *
                            (OneP + x2 * 0.0238095f *
                                        (OneN + x2 * 0.0138889f *
                                                    (OneP - x2 * 0.0090909f))));
  fTrig *= sign;

  if (flags & saturation_on)
    fTrig = esimd::saturate<float>(fTrig);

  return fTrig;
}

// scalar Input
template <typename T> ESIMD_INLINE float cos_emu(T x0, const uint flags) {
  simd<float, 1> x1;
  simd<float, 1> x2;
  simd<float, 1> t3;

  simd<float, 1> sign;
  simd<float, 1> fTrig;
  float TwoPI = CMPI * 2.0f;

  simd<float, 1> x = esimd::fmod(x0, TwoPI);

  simd<float, 1> CmpI(CMPI);
  simd<float, 1> OneP(1.f);
  simd<float, 1> OneN(-1.f);

  x1.merge(x - CMPI * 0.5f, CmpI * 1.5f - x, (x <= CMPI));
  x1.merge(CmpI * 0.5f - x, (x <= CMPI * 0.5f));
  x1.merge(x - CMPI * 1.5f, (x > CMPI * 1.5f));

  sign.merge(OneP, OneN, ((x < CMPI * 0.5f) | (x >= CMPI * 1.5f)));

  x2 = x1 * x1;
  t3 = x2 * x1 * 0.1666667f;
  fTrig =
      x1 + t3 * (OneN + x2 * 0.05f *
                            (OneP + x2 * 0.0238095f *
                                        (OneN + x2 * 0.0138889f *
                                                    (OneP - x2 * 0.0090909f))));
  fTrig *= sign;

  if (flags & saturation_on)
    fTrig = esimd::saturate<float>(fTrig);

  return fTrig[0];
}

namespace detail {

template <int N>
ESIMD_INLINE simd<float, N> tanh_cody_waite_impl(simd<float, N> x) {
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

  using RT = simd<float, N>;

  RT absX = esimd::abs(x);
  RT g = absX * absX;

  RT sign;
  sign.merge(-1.f, 1.f, x < 0.f);

  auto isLarge = absX > xlarge;
  auto minor = absX <= xlarge;
  auto isGtMed = minor & (absX > xmedium);
  auto isGtSmall = (absX > xsmall) & (absX <= xmedium);

  RT res;
  res.merge(sign, x, isLarge);
  auto temp = esimd::exp(absX * 2.0f * log2E) + 1.f;
  temp = ((temp - 2.f) / temp) * sign;
  res.merge(temp, isGtMed);
  res.merge((absX + absX * g * (g * p1 + p0) / (g + q0)) * sign, isGtSmall);

  return res;
}

template <int N> ESIMD_INLINE simd<float, N> tanh_impl(simd<float, N> x) {
  /*
   *      0                       x_small                          x_large
   * |    x    |  ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )  |      1
   *
   */

  constexpr float xsmall = 0.000045f; // same as exp(-10.0f)
  constexpr float xlarge = 88.f;
  constexpr float log2E = 1.442695f; // same as esimd::log(e)

  using RT = simd<float, N>;

  RT absX = esimd::abs(x);

  RT sign;
  sign.merge(-1.f, 1.f, x < 0.f);

  auto isLarge = (absX > xlarge);
  auto isLessE = (absX <= xlarge);

  RT res;
  res.merge(sign, x, isLarge);

  RT exp;
  exp = esimd::exp(absX * 2.f * log2E);

  res.merge(((exp - 1.f) / (exp + 1.f)) * sign, (absX > xsmall) & isLessE);

  return res;
}
} // namespace detail

/* tanh_cody_waite - Cody-Waite implementation for tanh(x) */
/* float input */
ESIMD_INLINE float tanh_cody_waite(float x) {
  return detail::tanh_cody_waite_impl(simd<float, 1>(x))[0];
}
/* vector input */
template <int N> ESIMD_INLINE simd<float, N> tanh_cody_waite(simd<float, N> x) {
  return detail::tanh_cody_waite_impl(x);
}

/* tanh - opencl like implementation for tanh(x) */
/* float input */
ESIMD_INLINE float tanh(float x) {
  return esimd::detail::tanh_impl(simd<float, 1>(x))[0];
}
/* vector input */
template <int N> ESIMD_INLINE simd<float, N> tanh(simd<float, N> x) {
  return esimd::detail::tanh_impl(x);
}

// reduction functions
namespace detail {

template <typename T0, typename T1, int SZ> struct esimd_apply_sum {
  template <typename... T>
  simd<T0, SZ> operator()(simd<T1, SZ> v1, simd<T1, SZ> v2) {
    return v1 + v2;
  }
};

template <typename T0, typename T1, int SZ> struct esimd_apply_prod {
  template <typename... T>
  simd<T0, SZ> operator()(simd<T1, SZ> v1, simd<T1, SZ> v2) {
    return v1 * v2;
  }
};

template <typename T0, typename T1, int SZ> struct esimd_apply_reduced_max {
  template <typename... T>
  simd<T0, SZ> operator()(simd<T1, SZ> v1, simd<T1, SZ> v2) {
    if constexpr (std::is_floating_point<T1>::value) {
      return __esimd_fmax<T1, SZ>(v1.data(), v2.data());
    } else if constexpr (std::is_unsigned<T1>::value) {
      return __esimd_umax<T1, SZ>(v1.data(), v2.data());
    } else {
      return __esimd_smax<T1, SZ>(v1.data(), v2.data());
    }
  }
};

template <typename T0, typename T1, int SZ> struct esimd_apply_reduced_min {
  template <typename... T>
  simd<T0, SZ> operator()(simd<T1, SZ> v1, simd<T1, SZ> v2) {
    if constexpr (std::is_floating_point<T1>::value) {
      return __esimd_fmin<T1, SZ>(v1.data(), v2.data());
    } else if constexpr (std::is_unsigned<T1>::value) {
      return __esimd_umin<T1, SZ>(v1.data(), v2.data());
    } else {
      return __esimd_smin<T1, SZ>(v1.data(), v2.data());
    }
  }
};

template <typename T0, typename T1, int SZ,
          template <typename RT, typename T, int N> class OpType>
T0 reduce_single(simd<T1, SZ> v) {
  if constexpr (SZ == 1) {
    return v[0];
  } else {
    static_assert(detail::isPowerOf2(SZ),
                  "Invaid input for reduce_single - the vector size must "
                  "be power of two.");
    constexpr int N = SZ / 2;
    simd<T0, N> tmp = OpType<T0, T1, N>()(v.template select<N, 1>(0),
                                          v.template select<N, 1>(N));
    return reduce_single<T0, T0, N, OpType>(tmp);
  }
}

template <typename T0, typename T1, int N1, int N2,
          template <typename RT, typename T, int N> class OpType>
T0 reduce_pair(simd<T1, N1> v1, simd<T1, N2> v2) {
  if constexpr (N1 == N2) {
    simd<T0, N1> tmp = OpType<T0, T1, N1>()(v1, v2);
    return reduce_single<T0, T0, N1, OpType>(tmp);
  } else if constexpr (N1 < N2) {
    simd<T0, N1> tmp1 = OpType<T0, T1, N1>()(v1, v2.template select<N1, 1>(0));
    constexpr int N = N2 - N1;
    using NT = simd<T0, N>;
    NT tmp2 = convert<T0>(v2.template select<N, 1>(N1).read());
    return reduce_pair<T0, T0, N1, N, OpType>(tmp1, tmp2);
  } else {
    static_assert(detail::isPowerOf2(N1),
                  "Invaid input for reduce_pair - N1 must be power of two.");
    constexpr int N = N1 / 2;
    simd<T0, N> tmp = OpType<T0, T1, N>()(v1.template select<N, 1>(0),
                                          v1.template select<N, 1>(N));
    using NT = simd<T0, N2>;
    NT tmp2 = convert<T0>(v2);
    return reduce_pair<T0, T0, N, N2, OpType>(tmp, tmp2);
  }
}

template <typename T0, typename T1, int SZ,
          template <typename RT, typename T, int N> class OpType>
T0 reduce(simd<T1, SZ> v) {
  constexpr bool isPowerOf2 = detail::isPowerOf2(SZ);
  if constexpr (isPowerOf2) {
    return reduce_single<T0, T1, SZ, OpType>(v);
  } else {
    constexpr unsigned N1 = 1u << detail::log2<SZ>();
    constexpr unsigned N2 = SZ - N1;

    simd<T1, N1> v1 = v.template select<N1, 1>(0);
    simd<T1, N2> v2 = v.template select<N2, 1>(N1);
    return reduce_pair<T0, T1, N1, N2, OpType>(v1, v2);
  }
};

template <typename T0, typename T1, int SZ>
ESIMD_INLINE ESIMD_NODEBUG T0 sum(simd<T1, SZ> v) {
  using TT = detail::computation_type_t<simd<T1, SZ>>;
  using RT = typename TT::element_type;
  T0 retv = reduce<RT, T1, SZ, esimd_apply_sum>(v);
  return retv;
}

template <typename T0, typename T1, int SZ>
ESIMD_INLINE ESIMD_NODEBUG T0 prod(simd<T1, SZ> v) {
  using TT = detail::computation_type_t<simd<T1, SZ>>;
  using RT = typename TT::element_type;
  T0 retv = reduce<RT, T1, SZ, esimd_apply_prod>(v);
  return retv;
}

} // namespace detail

template <typename T0, typename T1, int SZ>
ESIMD_INLINE ESIMD_NODEBUG T0 hmax(simd<T1, SZ> v) {
  T0 retv = detail::reduce<T1, T1, SZ, detail::esimd_apply_reduced_max>(v);
  return retv;
}

template <typename T0, typename T1, int SZ>
ESIMD_INLINE ESIMD_NODEBUG T0 hmin(simd<T1, SZ> v) {
  T0 retv = detail::reduce<T1, T1, SZ, detail::esimd_apply_reduced_min>(v);
  return retv;
}

template <typename T0, typename T1, int SZ, typename BinaryOperation>
ESIMD_INLINE ESIMD_NODEBUG T0 reduce(simd<T1, SZ> v, BinaryOperation op) {
  if constexpr (std::is_same<detail::remove_cvref_t<BinaryOperation>,
                             std::plus<>>::value) {
    T0 retv = detail::sum<T0>(v);
    return retv;
  } else if constexpr (std::is_same<detail::remove_cvref_t<BinaryOperation>,
                                    std::multiplies<>>::value) {
    T0 retv = detail::prod<T0>(v);
    return retv;
  }
}

template <typename T, int N> simd<T, N> dp4(simd<T, N> v1, simd<T, N> v2) {
  auto retv = __esimd_dp4<T, N>(v1.data(), v2.data());
  return retv;
}

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
