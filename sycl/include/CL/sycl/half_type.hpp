//==-------------- half_type.hpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/type_traits.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>

#ifdef __SYCL_DEVICE_ONLY__
// `constexpr` could work because the implicit conversion from `float` to
// `_Float16` can be `constexpr`.
#define __SYCL_CONSTEXPR_ON_DEVICE constexpr
#else
#define __SYCL_CONSTEXPR_ON_DEVICE
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
namespace host_half_impl {

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

} // namespace host_half_impl

namespace half_impl {
class half;

// Several aliases are defined below:
// - StorageT: actual representation of half data type. It is used by scalar
//   half values and by 'cl::sycl::vec' class. On device side, it points to some
//   native half data type, while on host some custom data type is used to
//   emulate operations of 16-bit floating-point values
//
// - BIsRepresentationT: data type which is used by built-in functions. It is
//   distinguished from StorageT, because on host, we can still operate on the
//   wrapper itself and there is no sense in direct usage of underlying data
//   type (too many changes required for BIs implementation without any
//   foreseeable profits)
//
// - VecNStorageT - representation of N-element vector of halfs. Follows the
//   same logic as StorageT
#ifdef __SYCL_DEVICE_ONLY__
  using StorageT = _Float16;
  using BIsRepresentationT = _Float16;

  using Vec2StorageT = StorageT __attribute__((ext_vector_type(2)));
  using Vec3StorageT = StorageT __attribute__((ext_vector_type(3)));
  using Vec4StorageT = StorageT __attribute__((ext_vector_type(4)));
  using Vec8StorageT = StorageT __attribute__((ext_vector_type(8)));
  using Vec16StorageT = StorageT __attribute__((ext_vector_type(16)));
#else
  using StorageT = detail::host_half_impl::half;
  // No need to extract underlying data type for built-in functions operating on
  // host
  using BIsRepresentationT = half;

  // On the host side we cannot use OpenCL cl_half# types as an underlying type
  // for vec because they are actually defined as an integer type under the
  // hood. As a result half values will be converted to the integer and passed
  // as a kernel argument which is expected to be floating point number.
  template <int NumElements> struct half_vec {
    alignas(detail::vector_alignment<StorageT, NumElements>::value)
      std::array<StorageT, NumElements> s;
  };

  using Vec2StorageT = half_vec<2>;
  using Vec3StorageT = half_vec<3>;
  using Vec4StorageT = half_vec<4>;
  using Vec8StorageT = half_vec<8>;
  using Vec16StorageT = half_vec<16>;
#endif

class half {
public:
  half() = default;
  half(const half &) = default;
  half(half &&) = default;

  __SYCL_CONSTEXPR_ON_DEVICE half(const float &rhs) : Data(rhs) {}

  half &operator=(const half &rhs) = default;

#ifndef __SYCL_DEVICE_ONLY__
  // Since StorageT and BIsRepresentationT are different on host, these two
  // helpers are required for 'vec' class
  half(const detail::host_half_impl::half &rhs) : Data(rhs) {};
  operator detail::host_half_impl::half() const { return Data; }
#endif // __SYCL_DEVICE_ONLY__

  // Operator +=, -=, *=, /=
  half &operator+=(const half &rhs) {
    Data += rhs.Data;
    return *this;
  }

  half &operator-=(const half &rhs) {
    Data -= rhs.Data;
    return *this;
  }

  half &operator*=(const half &rhs) {
    Data *= rhs.Data;
    return *this;
  }

  half &operator/=(const half &rhs) {
    Data /= rhs.Data;
    return *this;
  }

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
  operator float() const { return static_cast<float>(Data); }

  template <typename Key> friend struct std::hash;
private:
  StorageT Data;
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
} // __SYCL_INLINE_NAMESPACE(cl)

using half = cl::sycl::detail::half_impl::half;

// Partial specialization of some functions in namespace `std`
namespace std {


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

  static __SYCL_CONSTEXPR_ON_DEVICE const half(min)() noexcept {
    return SYCL_HLF_MIN;
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const half(max)() noexcept {
    return SYCL_HLF_MAX;
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const half lowest() noexcept {
    return -SYCL_HLF_MAX;
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const half epsilon() noexcept {
    return SYCL_HLF_EPSILON;
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const half round_error() noexcept {
    return 0.5F;
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const half infinity() noexcept {
    return __builtin_huge_valf();
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const half quiet_NaN() noexcept {
    return __builtin_nanf("");
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const half signaling_NaN() noexcept {
    return __builtin_nansf("");
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const half denorm_min() noexcept {
    return 5.96046e-08F;
  }
};

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

#undef __SYCL_CONSTEXPR_ON_DEVICE
