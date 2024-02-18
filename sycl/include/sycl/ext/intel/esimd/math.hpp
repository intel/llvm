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

#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/math_intrin.hpp>
#include <sycl/ext/intel/esimd/detail/operators.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/detail/util.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/simd_view.hpp>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd {

/// @addtogroup sycl_esimd_math
/// @{
/// @defgroup sycl_esimd_math_ext Hardware-accelerated math.
///
/// This is a group of APIs implementing standard math operations which are also
/// directly supported by the hardware. Usually the hardware support is a
/// specific message to the "extended math" GPU "shared function" unit, sent via
/// the \c math instruction. Most of the operations do not conform to OpenCL
/// requirements for accuracy, so should be used with care.
///
/// TODO Provide detailed spec of each operation.
/// @}

/// @addtogroup sycl_esimd_math
/// @{

/// Conversion of input vector elements of type \p T1 into vector of elements of
/// type \p T0 with saturation.
/// The following conversions are supported:
/// - \c T0 and \c T1 is the same floating-point type (including \c half). In
///   this case the result in the \c i'th lane is:
///     * \c 0 if \c src[i] is less than \c 0
///     * \c 1 if  \c src[i] is greater than \c 1
///     * src[i] otherwise
///
///    I.e. it is always a value in the range <code>[-1, 1]</code>.
/// - \c T0 is an integral type, \c T1 is any valid element type. In this case
///   the (per-element) result is the closest representable value. For example:
///     * Too big (exceeding representable range of \c T0) positive integral or
///       floating-point value src[i] of type \c T1 converted to \c T0
///       will result in <code>std:::numeric_limits<T0>::max()</code>.
///     * Too big negative value will be converted to
///       <code>std:::numeric_limits<T0>::min()</code>.
///     * Negative integer or floating point value converted to unsigned \c T1
///       will yield \c 0.
/// @tparam T0 Element type of the returned vector.
/// @tparam T1 Element type of the input vector.
/// @tparam SZ Size of the input and returned vector.
/// @param src The input vector.
/// @return Vector of \c src elements converted to \c T0 with saturation.
template <typename T0, typename T1, int SZ>
__ESIMD_API std::enable_if_t<!detail::is_generic_floating_point_v<T0> ||
                                 std::is_same_v<T1, T0>,
                             simd<T0, SZ>>
saturate(simd<T1, SZ> src) {
  if constexpr (detail::is_generic_floating_point_v<T0>)
    return __esimd_sat<T0, T1, SZ>(src.data());
  else if constexpr (detail::is_generic_floating_point_v<T1>) {
    if constexpr (std::is_unsigned_v<T0>)
      return __esimd_fptoui_sat<T0, T1, SZ>(src.data());
    else
      return __esimd_fptosi_sat<T0, T1, SZ>(src.data());
  } else if constexpr (std::is_unsigned_v<T0>) {
    if constexpr (std::is_unsigned_v<T1>)
      return __esimd_uutrunc_sat<T0, T1, SZ>(src.data());
    else
      return __esimd_ustrunc_sat<T0, T1, SZ>(src.data());
  } else {
    if constexpr (std::is_signed_v<T1>)
      return __esimd_sstrunc_sat<T0, T1, SZ>(src.data());
    else
      return __esimd_sutrunc_sat<T0, T1, SZ>(src.data());
  }
}

/// @cond ESIMD_DETAIL
// abs
namespace detail {

template <typename TRes, typename TArg, int SZ>
ESIMD_NODEBUG ESIMD_INLINE simd<TRes, SZ>
__esimd_abs_common_internal(simd<TArg, SZ> src0) {
  simd<TArg, SZ> Result = simd<TArg, SZ>(__esimd_abs<TArg, SZ>(src0.data()));
  return convert<TRes>(Result);
}

template <typename TRes, typename TArg>
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<detail::is_esimd_scalar<TRes>::value &&
                         detail::is_esimd_scalar<TArg>::value,
                     TRes>
    __esimd_abs_common_internal(TArg src0) {
  simd<TArg, 1> Src0 = src0;
  simd<TArg, 1> Result = __esimd_abs_common_internal<TArg>(Src0);
  return convert<TRes>(Result)[0];
}
} // namespace detail
/// @endcond ESIMD_DETAIL

/// Get absolute value (vector version)
/// @tparam TRes element type of the returned vector.
/// @tparam TArg element type of the input vector.
/// @tparam SZ size of the input and returned vector.
/// @param src0 the input vector.
/// @return vector of absolute values.
template <typename TRes, typename TArg, int SZ>
__ESIMD_API std::enable_if_t<
    !std::is_same<std::remove_const_t<TRes>, std::remove_const_t<TArg>>::value,
    simd<TRes, SZ>>
abs(simd<TArg, SZ> src0) {
  return detail::__esimd_abs_common_internal<TRes, TArg, SZ>(src0.data());
}

/// Get absolute value (scalar version)
/// @tparam T0 element type of the returned value.
/// @tparam T1 element type of the input value.
/// @param src0 the source operand.
/// @return absolute value.
template <typename TRes, typename TArg>
__ESIMD_API std::enable_if_t<!std::is_same<std::remove_const_t<TRes>,
                                           std::remove_const_t<TArg>>::value &&
                                 detail::is_esimd_scalar<TRes>::value &&
                                 detail::is_esimd_scalar<TArg>::value,
                             std::remove_const_t<TRes>>
abs(TArg src0) {
  return detail::__esimd_abs_common_internal<TRes, TArg>(src0);
}

/// Get absolute value (vector version). This is a specialization of a version
/// with three template parameters, where the element types of the input and
/// output vector are the same.
/// @tparam T1 element type of the input and output vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @return vector of absolute values.
template <typename T1, int SZ> __ESIMD_API simd<T1, SZ> abs(simd<T1, SZ> src0) {
  return detail::__esimd_abs_common_internal<T1, T1, SZ>(src0.data());
}

/// Get absolute value (scalar version). This is a specialization of a version
/// with two template parameters, where the types of the input and output value
/// are the same.
/// @tparam T1 element type of the input and output value.
/// @param src0 the source operand.
/// @return absolute value.
template <typename T1>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T1>::value,
                             std::remove_const_t<T1>>
abs(T1 src0) {
  return detail::__esimd_abs_common_internal<T1, T1>(src0);
}

/// Selects component-wise the maximum of the two vectors.
/// The source operands must be both of integer or both of floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.
template <typename T, int SZ, class Sat = saturation_off_tag>
__ESIMD_API simd<T, SZ>(max)(simd<T, SZ> src0, simd<T, SZ> src1, Sat sat = {}) {
  constexpr bool is_sat = std::is_same_v<Sat, saturation_on_tag>;

  if constexpr (std::is_floating_point<T>::value) {
    auto Result = __esimd_fmax<T, SZ>(src0.data(), src1.data());
    if constexpr (is_sat)
      Result = __esimd_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  } else if constexpr (std::is_unsigned<T>::value) {
    auto Result = __esimd_umax<T, SZ>(src0.data(), src1.data());
    if constexpr (is_sat)
      Result = __esimd_uutrunc_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  } else {
    auto Result = __esimd_smax<T, SZ>(src0.data(), src1.data());
    if constexpr (is_sat)
      Result = __esimd_sstrunc_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  }
}

/// Selects maximums for each element of the input vector and a scalar.
/// The source operands must be both of integer or both of
/// floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.
template <typename T, int SZ, class Sat = saturation_off_tag>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T>::value, simd<T, SZ>>(
    max)(simd<T, SZ> src0, T src1, Sat sat = {}) {
  simd<T, SZ> Src1 = src1;
  simd<T, SZ> Result = (esimd::max)(src0, Src1, sat);
  return Result;
}

/// Selects maximums for each element of the input scalar and a vector.
/// The source operands must be both of integer or both of
/// floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the scalar value.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.
template <typename T, int SZ, class Sat = saturation_off_tag>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T>::value, simd<T, SZ>>(
    max)(T src0, simd<T, SZ> src1, Sat sat = {}) {
  simd<T, SZ> Src0 = src0;
  simd<T, SZ> Result = (esimd::max)(Src0, src1, sat);
  return Result;
}

/// Selects maximum between two scalar values. (scalar version)
/// The source operands must be both of integer or both of floating-point type.
/// @tparam T element type of the input and return vectors.
/// @param src0 the scalar value.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return maximum value between the two inputs.
template <typename T, class Sat = saturation_off_tag>
ESIMD_NODEBUG ESIMD_INLINE
std::enable_if_t<detail::is_esimd_scalar<T>::value, T>(max)(T src0, T src1,
                                                            Sat sat = {}) {
  simd<T, 1> Src0 = src0;
  simd<T, 1> Src1 = src1;
  simd<T, 1> Result = (esimd::max)(Src0, Src1, sat);
  return Result[0];
}

/// Selects component-wise the minimum of the two vectors.
/// The source operands must be both of integer or both of floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.
template <typename T, int SZ, class Sat = saturation_off_tag>
__ESIMD_API simd<T, SZ>(min)(simd<T, SZ> src0, simd<T, SZ> src1, Sat sat = {}) {
  constexpr bool is_sat = std::is_same_v<Sat, saturation_on_tag>;

  if constexpr (std::is_floating_point<T>::value) {
    auto Result = __esimd_fmin<T, SZ>(src0.data(), src1.data());
    if constexpr (is_sat)
      Result = __esimd_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  } else if constexpr (std::is_unsigned<T>::value) {
    auto Result = __esimd_umin<T, SZ>(src0.data(), src1.data());
    if constexpr (is_sat)
      Result = __esimd_uutrunc_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  } else {
    auto Result = __esimd_smin<T, SZ>(src0.data(), src1.data());
    if constexpr (is_sat)
      Result = __esimd_sstrunc_sat<T, T, SZ>(Result);
    return simd<T, SZ>(Result);
  }
}

/// Selects minimums for each element of the input vector and a scalar.
/// The source operands must be both of integer or both of
/// floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.
template <typename T, int SZ, class Sat = saturation_off_tag>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T>::value, simd<T, SZ>>(
    min)(simd<T, SZ> src0, T src1, Sat sat = {}) {
  simd<T, SZ> Src1 = src1;
  simd<T, SZ> Result = (esimd::min)(src0, Src1, sat);
  return Result;
}

/// Selects minimums for each element of the input scalar and a vector.
/// The source operands must be both of integer or both of
/// floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the scalar value.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.
template <typename T, int SZ, class Sat = saturation_off_tag>
__ESIMD_API std::enable_if_t<detail::is_esimd_scalar<T>::value, simd<T, SZ>>(
    min)(T src0, simd<T, SZ> src1, Sat sat = {}) {
  simd<T, SZ> Src0 = src0;
  simd<T, SZ> Result = (esimd::min)(Src0, src1, sat);
  return Result;
}

/// Selects minimum between two scalar values.
/// The source operands must be both of integer or both of floating-point type.
/// @tparam T element type of the input and return vectors.
/// @param src0 the scalar value.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return minimum value between the two inputs.
template <typename T, class Sat = saturation_off_tag>
ESIMD_NODEBUG ESIMD_INLINE
std::enable_if_t<detail::is_esimd_scalar<T>::value, T>(min)(T src0, T src1,
                                                            Sat sat = {}) {
  simd<T, 1> Src0 = src0;
  simd<T, 1> Src1 = src1;
  simd<T, 1> Result = (esimd::min)(Src0, Src1, sat);
  return Result[0];
}

/// @} sycl_esimd_math

/// @addtogroup sycl_esimd_math_ext
/// @{

#define __ESIMD_UNARY_INTRINSIC_DEF(COND, name, iname)                         \
  /** Vector version.                                                       */ \
  template <class T, int N, class Sat = saturation_off_tag,                    \
            class = std::enable_if_t<COND>>                                    \
  __ESIMD_API simd<T, N> name(simd<T, N> src, Sat sat = {}) {                  \
    __ESIMD_DNS::vector_type_t<__ESIMD_DNS::__raw_t<T>, N> res =               \
        __esimd_##iname<T, N>(src.data());                                     \
    if constexpr (std::is_same_v<Sat, saturation_off_tag>)                     \
      return res;                                                              \
    else                                                                       \
      return esimd::saturate<T>(simd<T, N>(res));                              \
  }                                                                            \
                                                                               \
  /** Scalar version.                                                       */ \
  template <typename T, class Sat = saturation_off_tag,                        \
            class = std::enable_if_t<COND>>                                    \
  __ESIMD_API T name(T src, Sat sat = {}) {                                    \
    simd<T, 1> src_vec = src;                                                  \
    simd<T, 1> res = name<T, 1>(src_vec, sat);                                 \
    return res[0];                                                             \
  }

#define __ESIMD_EMATH_COND                                                     \
  detail::is_generic_floating_point_v<T> && (sizeof(T) <= 4)

#define __ESIMD_EMATH_IEEE_COND                                                \
  detail::is_generic_floating_point_v<T> && (sizeof(T) >= 4)

/// Inversion - calculates (1/x). Supports \c half and \c float.
/// Precision: 1 ULP.
__ESIMD_UNARY_INTRINSIC_DEF(__ESIMD_EMATH_COND, inv, inv)

/// Logarithm base 2. Supports \c half and \c float.
/// Precision depending on argument range:
/// - [0.5..2]: absolute error is <code>2^-21</code> or less
/// - (0..0.5) or (2..+INF]: relative error is  <code>2^-21</code> or less
__ESIMD_UNARY_INTRINSIC_DEF(__ESIMD_EMATH_COND, log2, log)

/// Exponent base 2. Supports \c half and \c float.
/// Precision: 4 ULP.
__ESIMD_UNARY_INTRINSIC_DEF(__ESIMD_EMATH_COND, exp2, exp)

/// Square root. Is not IEEE754-compatible.  Supports \c half and \c float.
/// Precision: 4 ULP.
__ESIMD_UNARY_INTRINSIC_DEF(__ESIMD_EMATH_COND, sqrt, sqrt)

/// IEEE754-compliant square root. Supports \c float and \c double.
__ESIMD_UNARY_INTRINSIC_DEF(__ESIMD_EMATH_IEEE_COND, sqrt_ieee, ieee_sqrt)

/// Square root reciprocal - calculates <code>1/sqrt(x)</code>.
/// Supports \c half and \c float.
/// Precision: 4 ULP.
__ESIMD_UNARY_INTRINSIC_DEF(__ESIMD_EMATH_COND, rsqrt, rsqrt)

/// Sine. Supports \c half and \c float.
/// Absolute error: \c 0.0008 or less for the range [-32767*pi, 32767*pi].
__ESIMD_UNARY_INTRINSIC_DEF(__ESIMD_EMATH_COND, sin, sin)

/// Cosine. Supports \c half and \c float.
/// Absolute error: \c 0.0008 or less for the range [-32767*pi, 32767*pi].
__ESIMD_UNARY_INTRINSIC_DEF(__ESIMD_EMATH_COND, cos, cos)

#undef __ESIMD_UNARY_INTRINSIC_DEF

#define __ESIMD_BINARY_INTRINSIC_DEF(COND, name, iname)                        \
  /** (vector, vector) version.                                             */ \
  template <class T, int N, class U, class Sat = saturation_off_tag,           \
            class = std::enable_if_t<COND>>                                    \
  __ESIMD_API simd<T, N> name(simd<T, N> src0, simd<U, N> src1,                \
                              Sat sat = {}) {                                  \
    using RawVecT = __ESIMD_DNS::vector_type_t<__ESIMD_DNS::__raw_t<T>, N>;    \
    RawVecT src1_raw_conv = detail::convert_vector<T, U, N>(src1.data());      \
    RawVecT res_raw = __esimd_##iname<T, N>(src0.data(), src1_raw_conv);       \
    if constexpr (std::is_same_v<Sat, saturation_off_tag>)                     \
      return res_raw;                                                          \
    else                                                                       \
      return esimd::saturate<T>(simd<T, N>(res_raw));                          \
  }                                                                            \
                                                                               \
  /** (vector, scalar) version.                                             */ \
  template <class T, int N, class U, class Sat = saturation_off_tag,           \
            class = std::enable_if_t<COND>>                                    \
  __ESIMD_API simd<T, N> name(simd<T, N> src0, U src1, Sat sat = {}) {         \
    return name<T, N, U>(src0, simd<U, N>(src1), sat);                         \
  }                                                                            \
                                                                               \
  /** (scalar, scalar) version.                                             */ \
  template <class T, class U, class Sat = saturation_off_tag,                  \
            class = std::enable_if_t<COND>>                                    \
  __ESIMD_API T name(T src0, U src1, Sat sat = {}) {                           \
    simd<T, 1> res = name<T, 1, U>(simd<T, 1>(src0), simd<U, 1>(src1), sat);   \
    return res[0];                                                             \
  }

/// Power - calculates \c src0 in power of \c src1. Note available in DG2, PVC.
///  Supports \c half and \c float.
/// TODO document accuracy etc.
__ESIMD_BINARY_INTRINSIC_DEF(__ESIMD_EMATH_COND, pow, pow)

/// IEEE754-compliant floating-point division. Supports \c float and \c double.
__ESIMD_BINARY_INTRINSIC_DEF(__ESIMD_EMATH_IEEE_COND, div_ieee, ieee_div)

#undef __ESIMD_BINARY_INTRINSIC_DEF
#undef __ESIMD_EMATH_COND
#undef __ESIMD_EMATH_IEEE_COND

/// @} sycl_esimd_math_ext

/// @addtogroup sycl_esimd_math
/// @{

/// @cond ESIMD_DETAIL
namespace detail {
// std::numbers::ln2_v<float> in c++20
constexpr float ln2 = 0.69314718f;
// std::numbers::log2e_v<float> in c++20
constexpr float log2e = 1.442695f;
} // namespace detail
/// @endcond ESIMD_DETAIL

/// Computes the natural logarithm of the given argument. This is an
/// emulated version based on the H/W supported log2.
/// @param the source operand to compute base-e logarithm of.
/// @return the base-e logarithm of \p src0.
template <class T, int SZ, class Sat = saturation_off_tag>
ESIMD_NODEBUG ESIMD_INLINE simd<T, SZ> log(simd<T, SZ> src0, Sat sat = {}) {
  using CppT = __ESIMD_DNS::__cpp_t<T>;
  simd<T, SZ> Result =
      esimd::log2<T, SZ, saturation_off_tag>(src0) * detail::ln2;

  if constexpr (std::is_same_v<Sat, saturation_off_tag>)
    return Result;
  else
    return esimd::saturate<T>(Result);
}

template <class T, class Sat = saturation_off_tag>
ESIMD_NODEBUG ESIMD_INLINE T log(T src0, Sat sat = {}) {
  return esimd::log<T, 1>(src0, sat)[0];
}

/// Computes e raised to the power of the given argument. This is an
/// emulated version based on the H/W supported exp2.
/// @param the source operand to compute base-e exponential of.
/// @return e raised to the power of \p src0.
template <class T, int SZ, class Sat = saturation_off_tag>
ESIMD_NODEBUG ESIMD_INLINE simd<T, SZ> exp(simd<T, SZ> src0, Sat sat = {}) {
  using CppT = __ESIMD_DNS::__cpp_t<T>;
  return esimd::exp2<T, SZ>(src0 * detail::log2e, sat);
}

template <class T, class Sat = saturation_off_tag>
ESIMD_NODEBUG ESIMD_INLINE T exp(T src0, Sat sat = {}) {
  return esimd::exp<T, 1>(src0, sat)[0];
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

/// @addtogroup sycl_esimd_bitmanip
/// @{

/// Pack a simd_mask into a single unsigned 32-bit integer value.
/// i'th bit in the returned value is set to the result of comparison of the
/// i'th element of the input argument to zero. "equals to zero" gives \c 0,
/// "not equal to zero" gives \c 1. Remaining (if any) bits if the result are
/// filled with \c 0.
/// @tparam N Size of the input mask.
/// @param src0 The input mask.
/// @return The packed mask as an <code>unsgined int</code> 32-bit value.
template <int N>
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<(N == 8 || N == 16 || N == 32), uint>
    pack_mask(simd_mask<N> src0) {
  return __esimd_pack_mask<N>(src0.data());
}

/// Unpack an unsigned 32-bit integer value into a simd_mask. Only \c N least
/// significant bits are used, where \c N is the number of elements in the
/// result mask. Each input bit is stored into the corresponding vector element
/// of the output mask.
/// @tparam N Size of the output mask.
/// @param src0 The input packed mask.
/// @return The unpacked mask as a simd_mask object.
template <int N>
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<(N == 8 || N == 16 || N == 32), simd_mask<N>>
    unpack_mask(uint src0) {
  return __esimd_unpack_mask<N>(src0);
}

/// @ref pack_mask specialization when the number of elements \c N is not \c 8,
/// \c 16 or \c 32.
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
__ESIMD_API std::enable_if_t<(std::is_same_v<T, ushort> ||
                              std::is_same_v<T, uint>)&&(N > 0 && N <= 32),
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
ESIMD_NODEBUG ESIMD_INLINE
    std::enable_if_t<std::is_integral<T>::value && (sizeof(T) <= 4),
                     simd<uint32_t, N>>
    cbit(simd<T, N> src) {
  return __esimd_cbit<T, N>(src.data());
}

/// Scalar version of \c cbit - both input and output are scalars rather
/// than vectors.
template <typename T>
__ESIMD_API
    std::enable_if_t<std::is_integral<T>::value && (sizeof(T) <= 4), uint32_t>
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

/// @} sycl_esimd_bitmanip

/// @addtogroup sycl_esimd_math
/// @{

/// \brief DP4A.
///
/// @param src0 the first source operand of dp4a operation.
///
/// @param src1 the second source operand of dp4a operation.
///
/// @param src2 the third source operand of dp4a operation.
///
/// @param sat saturation flag, which has default value of saturation_off.
///
/// Returns simd vector of the dp4a operation result.
///
template <typename T1, typename T2, typename T3, typename T4, int N,
          class Sat = saturation_off_tag>
__ESIMD_API std::enable_if_t<
    detail::is_dword_type<T1>::value && detail::is_dword_type<T2>::value &&
        detail::is_dword_type<T3>::value && detail::is_dword_type<T4>::value,
    simd<T1, N>>
dp4a(simd<T2, N> src0, simd<T3, N> src1, simd<T4, N> src2, Sat sat = {}) {
#if defined(__SYCL_DEVICE_ONLY__)
  simd<T1, N> Result;
  simd<T2, N> Src0 = src0;
  simd<T3, N> Src1 = src1;
  simd<T4, N> Src2 = src2;
  if constexpr (std::is_same_v<Sat, saturation_off_tag>) {
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
  return Result;
#else
  __ESIMD_UNSUPPORTED_ON_HOST;
#endif // __SYCL_DEVICE_ONLY__
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
/// @endcond ESIMD_DETAIL

/// Performs 'maximum' operation reduction over elements of the input vector,
/// that is, returns the maximal vector element.
/// @tparam T0 type of the return value.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input vector.
/// @param v the vector to perfrom reduction on
/// @return result of the reduction
template <typename T0, typename T1, int SZ>
ESIMD_INLINE ESIMD_NODEBUG T0 hmax(simd<T1, SZ> v) {
  T0 retv = detail::reduce<T1, T1, SZ, detail::esimd_apply_reduced_max>(v);
  return retv;
}

/// Performs 'minimum' operation reduction over elements of the input vector,
/// that is, returns the minimal vector element.
/// @tparam T0 type of the return value.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input vector.
/// @param v the vector to perfrom reduction on
/// @return result of the reduction
template <typename T0, typename T1, int SZ>
ESIMD_INLINE ESIMD_NODEBUG T0 hmin(simd<T1, SZ> v) {
  T0 retv = detail::reduce<T1, T1, SZ, detail::esimd_apply_reduced_min>(v);
  return retv;
}

/// Performs reduction over elements of the input vector.
/// @tparam T0 type of the return value.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input vector.
/// @tparam BinaryOperation type representing the operation. Can be an
///   instantion of one of the following types:
///   \li \c std::plus, performs addition operation
///   \li \c std::multiplies, performs multiplication operation
/// @param v the vector to perfrom reduction on
/// @param op reduction operation object, used to auto-deduce the
/// BinaryOperation
///   template parameter.
/// @return result of the reduction
// TODO 1) enforce BinaryOperation constraints 2) support std::minimum/maximum
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

/// @addtogroup sycl_esimd_logical
/// @{

/// This enum is used to encode all possible logical operations performed
/// on the 3 input operands. It is used as a template argument of the bfn()
/// function.
/// Example: d = bfn<~bfn_t::x & ~bfn_t::y & ~bfn_t::z>(s0, s1, s2);
enum class bfn_t : uint8_t { x = 0xAA, y = 0xCC, z = 0xF0 };

static constexpr bfn_t operator~(bfn_t x) {
  uint8_t val = static_cast<uint8_t>(x);
  uint8_t res = ~val;
  return static_cast<bfn_t>(res);
}

static constexpr bfn_t operator|(bfn_t x, bfn_t y) {
  uint8_t arg0 = static_cast<uint8_t>(x);
  uint8_t arg1 = static_cast<uint8_t>(y);
  uint8_t res = arg0 | arg1;
  return static_cast<bfn_t>(res);
}

static constexpr bfn_t operator&(bfn_t x, bfn_t y) {
  uint8_t arg0 = static_cast<uint8_t>(x);
  uint8_t arg1 = static_cast<uint8_t>(y);
  uint8_t res = arg0 & arg1;
  return static_cast<bfn_t>(res);
}

static constexpr bfn_t operator^(bfn_t x, bfn_t y) {
  uint8_t arg0 = static_cast<uint8_t>(x);
  uint8_t arg1 = static_cast<uint8_t>(y);
  uint8_t res = arg0 ^ arg1;
  return static_cast<bfn_t>(res);
}

/// Performs binary function computation with three vector operands.
/// @tparam FuncControl boolean function control expressed with bfn_t
/// enum values.
/// @tparam T type of the input vector element.
/// @tparam N size of the input vector.
/// @param s0 First boolean function argument.
/// @param s1 Second boolean function argument.
/// @param s2 Third boolean function argument.
template <bfn_t FuncControl, typename T, int N>
__ESIMD_API std::enable_if_t<std::is_integral_v<T>, __ESIMD_NS::simd<T, N>>
bfn(__ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
    __ESIMD_NS::simd<T, N> src2) {
  if constexpr ((sizeof(T) == 8) || ((sizeof(T) == 1) && (N % 4 == 0)) ||
                ((sizeof(T) == 2) && (N % 2 == 0))) {
    // Bitcast Nx8-byte vectors to 2xN vectors of 4-byte integers.
    // Bitcast Nx1-byte vectors to N/4 vectors of 4-byte integers.
    // Bitcast Nx2-byte vectors to N/2 vectors of 4-byte integers.
    auto Result = __ESIMD_NS::bfn<FuncControl>(
        src0.template bit_cast_view<int32_t>().read(),
        src1.template bit_cast_view<int32_t>().read(),
        src2.template bit_cast_view<int32_t>().read());
    return Result.template bit_cast_view<T>();
  } else if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
    constexpr uint8_t FC = static_cast<uint8_t>(FuncControl);
    return __esimd_bfn<FC, T, N>(src0.data(), src1.data(), src2.data());
  } else if constexpr (N % 2 == 0) {
    // Bitcast Nx1-byte vectors (N is even) to N/2 vectors of 2-byte integers.
    auto Result = __ESIMD_NS::bfn<FuncControl>(
        src0.template bit_cast_view<int16_t>().read(),
        src1.template bit_cast_view<int16_t>().read(),
        src2.template bit_cast_view<int16_t>().read());
    return Result.template bit_cast_view<T>();
  } else {
    // Odd number of 1-byte elements.
    __ESIMD_NS::simd<T, N + 1> Src0, Src1, Src2;
    Src0.template select<N, 1>() = src0;
    Src1.template select<N, 1>() = src1;
    Src2.template select<N, 1>() = src2;
    auto Result = __ESIMD_NS::bfn<FuncControl>(Src0, Src1, Src2);
    return Result.template select<N, 1>();
  }
}

/// Performs binary function computation with three scalar operands.
/// @tparam FuncControl boolean function control expressed with bfn_t enum
/// values.
/// @tparam T type of the input vector element.
/// @param s0 First boolean function argument.
/// @param s1 Second boolean function argument.
/// @param s2 Third boolean function argument.
template <bfn_t FuncControl, typename T>
ESIMD_NODEBUG ESIMD_INLINE std::enable_if_t<
    __ESIMD_DNS::is_esimd_scalar<T>::value && std::is_integral_v<T>, T>
bfn(T src0, T src1, T src2) {
  __ESIMD_NS::simd<T, 1> Src0 = src0;
  __ESIMD_NS::simd<T, 1> Src1 = src1;
  __ESIMD_NS::simd<T, 1> Src2 = src2;
  __ESIMD_NS::simd<T, 1> Result =
      esimd::bfn<FuncControl, T, 1>(Src0, Src1, Src2);
  return Result[0];
}

/// @} sycl_esimd_logical

/// Performs add with carry of 2 unsigned 32-bit vectors.
/// @tparam N size of the vectors
/// @param carry vector that is going to hold resulting carry flag
/// @param src0 first term
/// @param src1 second term
/// @return sum of 2 terms, carry flag is returned through \c carry parameter
template <int N>
__ESIMD_API __ESIMD_NS::simd<uint32_t, N>
addc(__ESIMD_NS::simd<uint32_t, N> &carry, __ESIMD_NS::simd<uint32_t, N> src0,
     __ESIMD_NS::simd<uint32_t, N> src1) {
  std::pair<__ESIMD_DNS::vector_type_t<uint32_t, N>,
            __ESIMD_DNS::vector_type_t<uint32_t, N>>
      Result = __esimd_addc<uint32_t, N>(src0.data(), src1.data());

  carry = Result.first;
  return Result.second;
}

/// Performs add with carry of a unsigned 32-bit vector and scalar.
/// @tparam N size of the vectors
/// @param carry vector that is going to hold resulting carry flag
/// @param src0 first term
/// @param src1 second term
/// @return sum of 2 terms, carry flag is returned through \c carry parameter
template <int N>
__ESIMD_API __ESIMD_NS::simd<uint32_t, N>
addc(__ESIMD_NS::simd<uint32_t, N> &carry, __ESIMD_NS::simd<uint32_t, N> src0,
     uint32_t src1) {
  __ESIMD_NS::simd<uint32_t, N> Src1V = src1;
  return addc(carry, src0, Src1V);
}

/// Performs add with carry of a unsigned 32-bit scalar and vector.
/// @tparam N size of the vectors
/// @param carry vector that is going to hold resulting carry flag
/// @param src0 first term
/// @param src1 second term
/// @return sum of 2 terms, carry flag is returned through \c carry parameter
template <int N>
__ESIMD_API __ESIMD_NS::simd<uint32_t, N>
addc(__ESIMD_NS::simd<uint32_t, N> &carry, uint32_t src0,
     __ESIMD_NS::simd<uint32_t, N> src1) {
  __ESIMD_NS::simd<uint32_t, N> Src0V = src0;
  return addc(carry, Src0V, src1);
}

/// Performs add with carry of a unsigned 32-bit scalars.
/// @tparam N size of the vectors
/// @param carry scalar that is going to hold resulting carry flag
/// @param src0 first term
/// @param src1 second term
/// @return sum of 2 terms, carry flag is returned through \c carry parameter
__ESIMD_API uint32_t addc(uint32_t &carry, uint32_t src0, uint32_t src1) {
  __ESIMD_NS::simd<uint32_t, 1> CarryV = carry;
  __ESIMD_NS::simd<uint32_t, 1> Src0V = src0;
  __ESIMD_NS::simd<uint32_t, 1> Src1V = src1;
  __ESIMD_NS::simd<uint32_t, 1> Res = addc(CarryV, Src0V, Src1V);
  carry = CarryV[0];
  return Res[0];
}

/// Performs substraction with borrow of 2 unsigned 32-bit vectors.
/// @tparam N size of the vectors
/// @param borrow vector that is going to hold resulting borrow flag
/// @param src0 first term
/// @param src1 second term
/// @return difference of 2 terms, borrow flag is returned through \c borrow
/// parameter
template <int N>
__ESIMD_API __ESIMD_NS::simd<uint32_t, N>
subb(__ESIMD_NS::simd<uint32_t, N> &borrow, __ESIMD_NS::simd<uint32_t, N> src0,
     __ESIMD_NS::simd<uint32_t, N> src1) {
  std::pair<__ESIMD_DNS::vector_type_t<uint32_t, N>,
            __ESIMD_DNS::vector_type_t<uint32_t, N>>
      Result = __esimd_subb<uint32_t, N>(src0.data(), src1.data());

  borrow = Result.first;
  return Result.second;
}

/// Performs substraction with borrow of unsigned 32-bit vector and scalar.
/// @tparam N size of the vectors
/// @param borrow vector that is going to hold resulting borrow flag
/// @param src0 first term
/// @param src1 second term
/// @return difference of 2 terms, borrow flag is returned through \c borrow
/// parameter
template <int N>
__ESIMD_API __ESIMD_NS::simd<uint32_t, N>
subb(__ESIMD_NS::simd<uint32_t, N> &borrow, __ESIMD_NS::simd<uint32_t, N> src0,
     uint32_t src1) {
  __ESIMD_NS::simd<uint32_t, N> Src1V = src1;
  return subb(borrow, src0, Src1V);
}

/// Performs substraction with borrow of unsigned 32-bit scalar and vector.
/// @tparam N size of the vectors
/// @param borrow vector that is going to hold resulting borrow flag
/// @param src0 first term
/// @param src1 second term
/// @return difference of 2 terms, borrow flag is returned through \c borrow
/// parameter
template <int N>
__ESIMD_API __ESIMD_NS::simd<uint32_t, N>
subb(__ESIMD_NS::simd<uint32_t, N> &borrow, uint32_t src0,
     __ESIMD_NS::simd<uint32_t, N> src1) {
  __ESIMD_NS::simd<uint32_t, N> Src0V = src0;
  return subb(borrow, Src0V, src1);
}

/// Performs substraction with borrow of 2 unsigned 32-bit scalars.
/// @tparam N size of the vectors
/// @param borrow scalar that is going to hold resulting borrow flag
/// @param src0 first term
/// @param src1 second term
/// @return difference of 2 terms, borrow flag is returned through \c borrow
/// parameter
__ESIMD_API uint32_t subb(uint32_t &borrow, uint32_t src0, uint32_t src1) {
  __ESIMD_NS::simd<uint32_t, 1> BorrowV = borrow;
  __ESIMD_NS::simd<uint32_t, 1> Src0V = src0;
  __ESIMD_NS::simd<uint32_t, 1> Src1V = src1;
  __ESIMD_NS::simd<uint32_t, 1> Res = subb(BorrowV, Src0V, Src1V);
  borrow = BorrowV[0];
  return Res[0];
}

/// @} sycl_esimd_math

} // namespace ext::intel::esimd
} // namespace _V1
} // namespace sycl
