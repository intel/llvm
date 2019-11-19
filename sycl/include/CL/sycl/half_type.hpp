//==-------------- half_type.hpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>

namespace cl {
namespace sycl {
namespace detail {
namespace half_impl {

class half {
public:
  half() = default;
  half(const half &) = default;
  half(half &&) = default;

  half(const float &rhs);

  half &operator=(const half &rhs) = default;

  // Operator +=, -=, *=, /=
  half &operator+=(const half &rhs);

  half &operator-=(const half &rhs);

  half &operator*=(const half &rhs);

  half &operator/=(const half &rhs);

  // Operator ++, --
  half &operator++() {
    *this += 1;
    return *this;
  }

  half operator++(int) {
    half ret(*this);
    operator++();
    return ret;
  }

  half &operator--() {
    *this -= 1;
    return *this;
  }

  half operator--(int) {
    half ret(*this);
    operator--();
    return ret;
  }

  // Operator float
  operator float() const;

  template <typename Key> friend struct std::hash;

private:
  uint16_t Buf;
};
} // namespace half_impl

// Accroding to C++ standard math functions from cmath/math.h should work only
// on arithmetic types. We can't specify half type as arithmetic/floating
// point(via std::is_floating_point) since only float, double and long double
// types are "floating point" according to the standard. In order to use half
// type with these math functions we cast half to float using template
// function helper.
template <typename T> inline T cast_if_host_half(T val) { return val; }

inline float cast_if_host_half(half_impl::half val) {
  return static_cast<float>(val);
}

} // namespace detail

} // namespace sycl
} // namespace cl

#ifdef __SYCL_DEVICE_ONLY__
using half = _Float16;
#else
using half = cl::sycl::detail::half_impl::half;
#endif

// Partial specialization of some functions in namespace `std`
namespace std {

#ifdef __SYCL_DEVICE_ONLY__
// `constexpr` could work because the implicit conversion from `float` to
// `_Float16` can be `constexpr`.
#define CONSTEXPR_QUALIFIER constexpr
#else
// The qualifier is `const` instead of `constexpr` that is original to be
// because the constructor is not `constexpr` function.
#define CONSTEXPR_QUALIFIER const
#endif

// Partial specialization of `std::hash<cl::sycl::half>`
template <> struct hash<half> {
  size_t operator()(half const &Key) const noexcept {
    return hash<uint16_t>{}(reinterpret_cast<const uint16_t &>(Key));
  }
};

// Partial specialization of `std::numeric<cl::sycl::half>`

// All following values are either calculated based on description of each
// function/value on https://en.cppreference.com/w/cpp/types/numeric_limits, or
// cl_platform.h.
#define SYCL_HLF_MIN 6.103515625e-05F

#define SYCL_HLF_MAX 65504.0F

#define SYCL_HLF_MAX_10_EXP 4

#define SYCL_HLF_MAX_EXP 16

#define SYCL_HLF_MIN_10_EXP (-4)

#define SYCL_HLF_MIN_EXP (-13)

#define SYCL_HLF_MANT_DIG 11

#define SYCL_HLF_DIG 3

#define SYCL_HLF_DECIMAL_DIG 5

#define SYCL_HLF_EPSILON 9.765625e-04F

#define SYCL_HLF_RADIX 2

template <> struct numeric_limits<half> {
  static constexpr const bool is_specialized = true;

  static constexpr const bool is_signed = true;

  static constexpr const bool is_integer = false;

  static constexpr const bool is_exact = false;

  static constexpr const bool has_infinity = true;

  static constexpr const bool has_quiet_NaN = true;

  static constexpr const bool has_signaling_NaN = true;

  static constexpr const float_denorm_style has_denorm = denorm_present;

  static constexpr const bool has_denorm_loss = false;

  static constexpr const bool tinyness_before = false;

  static constexpr const bool traps = false;

  static constexpr const int max_exponent10 = SYCL_HLF_MAX_10_EXP;

  static constexpr const int max_exponent = SYCL_HLF_MAX_EXP;

  static constexpr const int min_exponent10 = SYCL_HLF_MIN_10_EXP;

  static constexpr const int min_exponent = SYCL_HLF_MIN_EXP;

  static constexpr const int radix = SYCL_HLF_RADIX;

  static constexpr const int max_digits10 = SYCL_HLF_DECIMAL_DIG;

  static constexpr const int digits = SYCL_HLF_MANT_DIG;

  static constexpr const bool is_bounded = true;

  static constexpr const int digits10 = SYCL_HLF_DIG;

  static constexpr const bool is_modulo = false;

  static constexpr const bool is_iec559 = true;

  static constexpr const float_round_style round_style = round_to_nearest;

  static CONSTEXPR_QUALIFIER half min() noexcept { return SYCL_HLF_MIN; }

  static CONSTEXPR_QUALIFIER half max() noexcept { return SYCL_HLF_MAX; }

  static CONSTEXPR_QUALIFIER half lowest() noexcept { return -SYCL_HLF_MAX; }

  static CONSTEXPR_QUALIFIER half epsilon() noexcept {
    return SYCL_HLF_EPSILON;
  }

  static CONSTEXPR_QUALIFIER half round_error() noexcept { return 0.5F; }

  static CONSTEXPR_QUALIFIER half infinity() noexcept {
    return __builtin_huge_valf();
  }

  static CONSTEXPR_QUALIFIER half quiet_NaN() noexcept {
    return __builtin_nanf("");
  }

  static CONSTEXPR_QUALIFIER half signaling_NaN() noexcept {
    return __builtin_nansf("");
  }

  static CONSTEXPR_QUALIFIER half denorm_min() noexcept { return 5.96046e-08F; }
};

#undef CONSTEXPR_QUALIFIER

} // namespace std

inline std::ostream &operator<<(std::ostream &O, half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

inline std::istream &operator>>(std::istream &I, half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}
