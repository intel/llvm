//==----------- half_type_impl.hpp - SYCL half type implementation ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/bit_cast.hpp> // for bit_cast
#include <sycl/detail/defines.hpp>
#include <sycl/detail/fwd/half.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/aspects.hpp>
#endif

#include <cstdint>
#include <limits> // for float_denorm_style, float_round_style
#include <type_traits>

#ifndef __FLT16_MANT_DIG__
#error "sycl::half requires a host compiler and target with _Float16 support."
#endif

namespace sycl {
inline namespace _V1 {

namespace ext::intel::esimd::detail {
class WrapperElementTypeProxy;
} // namespace ext::intel::esimd::detail

namespace detail {

inline constexpr uint16_t float2Half(const float &Val) {
  return sycl::bit_cast<uint16_t>(static_cast<_Float16>(Val));
}

inline constexpr float half2Float(const uint16_t &Val) {
  return static_cast<float>(sycl::bit_cast<_Float16>(Val));
}

// Type in which a mixed half/T operation is computed: T if it is a wider
// floating-point type, NarrowT otherwise.
template <typename T, typename NarrowT>
using HalfMixedT =
    std::conditional_t<std::is_floating_point_v<T> && (sizeof(T) > 2), T,
                       NarrowT>;

namespace half_impl {
class half;

#ifndef __SYCL_DEVICE_ONLY__
class half {
#else
class [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] half {
#endif
public:
  half() = default;
  constexpr half(const half &) = default;
  constexpr half(half &&) = default;
  constexpr half &operator=(const half &) = default;
  constexpr half &operator=(half &&) = default;

  constexpr half(float Value) : Data(static_cast<StorageT>(Value)) {}

  constexpr operator float() const { return static_cast<float>(Data); }

  constexpr half &operator++() {
    ++Data;
    return *this;
  }
  constexpr half operator++(int) {
    half Old(*this);
    ++Data;
    return Old;
  }
  constexpr half &operator--() {
    --Data;
    return *this;
  }
  constexpr half operator--(int) {
    half Old(*this);
    --Data;
    return Old;
  }

  friend constexpr half operator+(half rhs) { return rhs; }
  friend constexpr half operator-(half rhs) { return fromNative(-rhs.Data); }

  // The mixed operators are overloads for exact types rather than a template,
  // so that operands that convert implicitly to these types (enumerations,
  // sycl::id<1>, ...) are accepted, as by the built-in operators.
#define _SYCL_HALF_MIXED_TYPES(X, op)                                          \
  X(op, int)                                                                   \
  X(op, unsigned int)                                                          \
  X(op, long)                                                                  \
  X(op, unsigned long)                                                         \
  X(op, long long)                                                             \
  X(op, unsigned long long)                                                    \
  X(op, float)                                                                 \
  X(op, double)                                                                \
  X(op, long double)

#define _SYCL_GEN_MIXED_OPT(op, T)                                             \
  friend constexpr detail::HalfMixedT<T, half> operator op(half lhs, T rhs) {  \
    using OpT = detail::HalfMixedT<T, StorageT>;                               \
    return fromNative(static_cast<OpT>(lhs.Data) op static_cast<OpT>(rhs));    \
  }                                                                            \
  friend constexpr detail::HalfMixedT<T, half> operator op(T lhs, half rhs) {  \
    using OpT = detail::HalfMixedT<T, StorageT>;                               \
    return fromNative(static_cast<OpT>(lhs) op static_cast<OpT>(rhs.Data));    \
  }

#define _SYCL_GEN_OPT(op)                                                      \
  friend constexpr half operator op(half lhs, half rhs) {                      \
    return fromNative(lhs.Data op rhs.Data);                                   \
  }                                                                            \
  _SYCL_HALF_MIXED_TYPES(_SYCL_GEN_MIXED_OPT, op)

  _SYCL_GEN_OPT(+)
  _SYCL_GEN_OPT(-)
  _SYCL_GEN_OPT(*)
  _SYCL_GEN_OPT(/)

#undef _SYCL_GEN_OPT
#undef _SYCL_GEN_MIXED_OPT

  // Compound assignment converts the other operand to half first, so that
  // double operands such as literals don't make device code depend on
  // aspect::fp64.
  // Class operands that convert to half, such as bfloat16, go through that
  // conversion: their conversions to other arithmetic types would make the
  // exact-type overloads ambiguous.
#define _SYCL_GEN_MIXED_OPT(op, T)                                             \
  constexpr half &operator op(T rhs) {                                         \
    Data op static_cast<StorageT>(rhs);                                        \
    return *this;                                                              \
  }

#define _SYCL_GEN_OPT(op)                                                      \
  constexpr half &operator op(half rhs) {                                      \
    Data op rhs.Data;                                                          \
    return *this;                                                              \
  }                                                                            \
  template <typename T,                                                        \
            std::enable_if_t<std::is_class_v<T> &&                             \
                                 std::is_convertible_v<const T &, half>,       \
                             bool> = true>                                     \
  constexpr half &operator op(const T &rhs) {                                  \
    return *this op static_cast<half>(rhs);                                    \
  }                                                                            \
  _SYCL_HALF_MIXED_TYPES(_SYCL_GEN_MIXED_OPT, op)

  _SYCL_GEN_OPT(+=)
  _SYCL_GEN_OPT(-=)
  _SYCL_GEN_OPT(*=)
  _SYCL_GEN_OPT(/=)

#undef _SYCL_GEN_OPT
#undef _SYCL_GEN_MIXED_OPT

  // Integers compare as float, so that values outside the half range don't
  // round to infinity.
#define _SYCL_GEN_MIXED_OPT(op, T)                                             \
  friend constexpr bool operator op(half lhs, T rhs) {                         \
    using OpT = detail::HalfMixedT<T, float>;                                  \
    return static_cast<OpT>(lhs.Data) op static_cast<OpT>(rhs);                \
  }                                                                            \
  friend constexpr bool operator op(T lhs, half rhs) {                         \
    using OpT = detail::HalfMixedT<T, float>;                                  \
    return static_cast<OpT>(lhs) op static_cast<OpT>(rhs.Data);                \
  }

#define _SYCL_GEN_OPT(op)                                                      \
  friend constexpr bool operator op(half lhs, half rhs) {                      \
    return lhs.Data op rhs.Data;                                               \
  }                                                                            \
  _SYCL_HALF_MIXED_TYPES(_SYCL_GEN_MIXED_OPT, op)

  _SYCL_GEN_OPT(==)
  _SYCL_GEN_OPT(!=)
  _SYCL_GEN_OPT(<)
  _SYCL_GEN_OPT(>)
  _SYCL_GEN_OPT(<=)
  _SYCL_GEN_OPT(>=)

#undef _SYCL_GEN_OPT
#undef _SYCL_GEN_MIXED_OPT
#undef _SYCL_HALF_MIXED_TYPES

  friend class sycl::ext::intel::esimd::detail::WrapperElementTypeProxy;

private:
  // Wraps a _Float16 result without a round trip through float.
  template <typename T>
  static constexpr std::conditional_t<std::is_same_v<T, StorageT>, half, T>
  fromNative(T Value) {
    if constexpr (std::is_same_v<T, StorageT>) {
      half Result{};
      Result.Data = Value;
      return Result;
    } else {
      return Value;
    }
  }

  friend constexpr half CreateHostHalfRaw(uint16_t X);
  friend std::numeric_limits<half>;

  StorageT Data;
};

constexpr inline half CreateHostHalfRaw(uint16_t X) {
  return half::fromNative(sycl::bit_cast<StorageT>(X));
}
} // namespace half_impl

// According to the C++ standard, math functions from cmath/math.h should work
// only on arithmetic types. We can't specify half type as arithmetic/floating
// point(via std::is_floating_point) since only float, double and long double
// types are "floating point" according to the standard. In order to use half
// type with these math functions we cast half to float using template
// function helper.
template <typename T> inline T cast_if_host_half(T val) { return val; }

inline float cast_if_host_half(half_impl::half val) {
  return static_cast<float>(val);
}

} // namespace detail

} // namespace _V1
} // namespace sycl

// Partial specialization of some functions in namespace `std`
namespace std {

template <> struct numeric_limits<sycl::half> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr bool tinyness_before = false;
  static constexpr bool traps = false;
  static constexpr int max_exponent10 = __FLT16_MAX_10_EXP__;
  static constexpr int max_exponent = __FLT16_MAX_EXP__;
  static constexpr int min_exponent10 = __FLT16_MIN_10_EXP__;
  static constexpr int min_exponent = __FLT16_MIN_EXP__;
  static constexpr int radix = 2;
  static constexpr int max_digits10 = __FLT16_DECIMAL_DIG__;
  static constexpr int digits = __FLT16_MANT_DIG__;
  static constexpr bool is_bounded = true;
  static constexpr int digits10 = __FLT16_DIG__;
  static constexpr bool is_modulo = false;
  static constexpr bool is_iec559 = true;
  static constexpr float_round_style round_style = round_to_nearest;

  // The values are built as _Float16 without a round trip through float: a
  // signaling NaN converted from float would become quiet. Finite values are
  // exact float literals because the F16 suffix requires C++23.
  static constexpr sycl::half(min)() noexcept {
    return sycl::half::fromNative(
        static_cast<sycl::detail::half_impl::StorageT>(0x1p-14f));
  }
  static constexpr sycl::half(max)() noexcept {
    return sycl::half::fromNative(
        static_cast<sycl::detail::half_impl::StorageT>(0x1.ffcp15f));
  }
  static constexpr sycl::half lowest() noexcept {
    return sycl::half::fromNative(
        static_cast<sycl::detail::half_impl::StorageT>(-0x1.ffcp15f));
  }
  static constexpr sycl::half epsilon() noexcept {
    return sycl::half::fromNative(
        static_cast<sycl::detail::half_impl::StorageT>(0x1p-10f));
  }
  static constexpr sycl::half round_error() noexcept {
    return sycl::half::fromNative(
        static_cast<sycl::detail::half_impl::StorageT>(0.5f));
  }
  static constexpr sycl::half infinity() noexcept {
    return sycl::half::fromNative(__builtin_inff16());
  }
  static constexpr sycl::half quiet_NaN() noexcept {
    return sycl::half::fromNative(__builtin_nanf16(""));
  }
  static constexpr sycl::half signaling_NaN() noexcept {
    return sycl::half::fromNative(__builtin_nansf16(""));
  }
  static constexpr sycl::half denorm_min() noexcept {
    return sycl::half::fromNative(
        static_cast<sycl::detail::half_impl::StorageT>(0x1p-24f));
  }
};

} // namespace std
