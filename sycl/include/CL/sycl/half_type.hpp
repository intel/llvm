//==-------------- half_type.hpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/export.hpp>
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

#ifdef _MSC_VER
// This feature is not supported in MSVC.
#define __builtin_expect(a, b) (a)
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

constexpr uint16_t float2Half(const float &Val) {
  // First part of the calculations - get the Exponent and Fractional
  // Get bool sign of Val and its absolute value for calculations
  bool BSign = Val < 0.0f;
  float AbsVal = BSign ? -Val : Val;
  int MeaninglessBits = 0;
  int Exponent = 0;
  // Fractional part
  uint64_t FracVal = 0;

  if (Val != 0) {
    if (AbsVal == std::numeric_limits<float>::infinity()) {
      Exponent = 255;
    } else if (std::isnan(AbsVal)) {
      Exponent = 0xff;
      // Change the first bit to 1 for NaN check below
      FracVal = 0x8000000000000000;
    } else {
      // The main idea is to bring float value first to integer
      // Second bring it to range [0, 2^64-1] that suit to uint_64
      // For that it needs to up the number to values >= 2^87
      // Then downscale the number to the desirable range

      // The number will be integer anyway if it >= 2^23
      // If the num >= 2^87 then if we downscale it by 2^64 we get
      // integer number
      Exponent = 254;
      while (AbsVal < 0x1p87f) {
        // We upscale the value by 2^41 to finally get number < 2^128
        AbsVal *= 0x1p41f;
        Exponent -= 41;
      }

      // Downscale value to range [0, 2^64-1] by 2^-64
      // The number is integer and it will be converted to uint_64t
      // with all the bits of the fractional part
      FracVal = (uint64_t)(AbsVal * 0x1p-64f);

      // Next it needs to be count of bits that are not frac part
      // It is needed for get the correct exponent from initial Val
      bool Buf = 0;

      // The meaningless bits are ended with the first 1
      while (!Buf) {
        Buf = (FracVal >> (63 - MeaninglessBits)) & 1;
        MeaninglessBits++;
      }
      // Gets the correct exponent
      Exponent -= (MeaninglessBits - 1);

      // If the float value is denormalized, then exponent = 0
      // And the first 8 bits are meaningless
      if (Exponent <= 0) {
        Exponent = 0;
        MeaninglessBits = 8;
      }
    }
  }
  // Second part of the calculations - get the half value

  // Extract the sign from the bool value
  const uint16_t Sign = 0x0001 & BSign;
  // Extract the fraction from the FracValue
  const uint32_t Frac32 = (FracVal << (MeaninglessBits)) >> (64 - 23);
  // Extract the exponent from the int exponent
  const uint8_t Exp32 = Exponent;
  const int16_t Exp32Diff = Exp32 - 127;

  // intialize to 0, covers the case for 0 and small numbers
  uint16_t Exp16 = 0, Frac16 = 0;

  if (__builtin_expect(Exp32Diff > 15, 0)) {
    // Infinity and big numbers convert to infinity
    Exp16 = 0x1f;
  } else if (__builtin_expect(Exp32Diff > -14, 0)) {
    // normal range for half type
    Exp16 = Exp32Diff + 15;
    // convert 23-bit mantissa to 10-bit mantissa.
    Frac16 = Frac32 >> 13;
    // Round the mantissa as given in OpenCL spec section : 6.1.1.1 The half
    // data type.
    if (Frac32 >> 12 & 0x01)
      Frac16 += 1;
  } else if (__builtin_expect(Exp32Diff > -24, 0)) {
    // subnormals
    Frac16 = (Frac32 | (uint32_t(1) << 23)) >> (-Exp32Diff - 1);
  }

  if (__builtin_expect(Exp32 == 0xff && Frac32 != 0, 0)) {
    // corner case: FP32 is NaN
    Exp16 = 0x1F;
    Frac16 = 0x200;
  }

  // Compose the final FP16 binary
  uint16_t Ret = 0;
  Ret |= Sign;
  Ret |= Exp16 << 10;
  Ret += Frac16; // Add the carry bit from operation Frac16 += 1;

  return Ret;
}
namespace host_half_impl {

class __SYCL_EXPORT half {
public:
  half() = default;
  constexpr half(const half &) = default;
  constexpr half(half &&) = default;

  constexpr half(const float &rhs) : Buf(float2Half(rhs)) {}

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

  // Operator neg
  constexpr half &operator-() {
    Buf ^= 0x8000;
    return *this;
  }

  // Operator float
  operator float() const;

  template <typename Key> friend struct std::hash;

  // Initialize underlying data
  constexpr explicit half(uint16_t x) : Buf(x) {}

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
  constexpr half(const half &) = default;
  constexpr half(half &&) = default;

  constexpr half(const float &rhs) : Data(rhs) {}

  half &operator=(const half &rhs) = default;

#ifndef __SYCL_DEVICE_ONLY__
  // Since StorageT and BIsRepresentationT are different on host, these two
  // helpers are required for 'vec' class
  constexpr half(const detail::host_half_impl::half &rhs) : Data(rhs){};
  constexpr operator detail::host_half_impl::half() const { return Data; }
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
  constexpr half &operator-() {
    Data = -Data;
    return *this;
  }
  constexpr half operator-() const {
    half r = *this;
    return -r;
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

// Partial specialization of some functions in namespace `std`
namespace std {

// Partial specialization of `std::hash<cl::sycl::half>`
template <> struct hash<cl::sycl::half> {
  size_t operator()(cl::sycl::half const &Key) const noexcept {
    return hash<uint16_t>{}(reinterpret_cast<const uint16_t &>(Key));
  }
};

// Partial specialization of `std::numeric<cl::sycl::half>`
template <> struct numeric_limits<cl::sycl::half> {
  // All following values are either calculated based on description of each
  // function/value on https://en.cppreference.com/w/cpp/types/numeric_limits,
  // or cl_platform.h.
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
  static constexpr int max_exponent10 = 4;
  static constexpr int max_exponent = 16;
  static constexpr int min_exponent10 = -4;
  static constexpr int min_exponent = -13;
  static constexpr int radix = 2;
  static constexpr int max_digits10 = 5;
  static constexpr int digits = 11;
  static constexpr bool is_bounded = true;
  static constexpr int digits10 = 3;
  static constexpr bool is_modulo = false;
  static constexpr bool is_iec559 = true;
  static constexpr float_round_style round_style = round_to_nearest;

  static __SYCL_CONSTEXPR_ON_DEVICE const cl::sycl::half(min)() noexcept {
    return 6.103515625e-05f; // half minimum value
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const cl::sycl::half(max)() noexcept {
    return 65504.0f; // half maximum value
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const cl::sycl::half lowest() noexcept {
    return -65504.0f; // -1*(half maximum value)
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const cl::sycl::half epsilon() noexcept {
    return 9.765625e-04f; // half epsilon
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const cl::sycl::half
  round_error() noexcept {
    return 0.5f;
  }

  static constexpr const cl::sycl::half infinity() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_huge_valf();
#else
    return cl::sycl::detail::host_half_impl::half(
        static_cast<uint16_t>(0x7C00));
#endif
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const cl::sycl::half quiet_NaN() noexcept {
    return __builtin_nanf("");
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const cl::sycl::half
  signaling_NaN() noexcept {
    return __builtin_nansf("");
  }

  static __SYCL_CONSTEXPR_ON_DEVICE const cl::sycl::half denorm_min() noexcept {
    return 5.96046e-08f;
  }
};

} // namespace std

inline std::ostream &operator<<(std::ostream &O, cl::sycl::half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

inline std::istream &operator>>(std::istream &I, cl::sycl::half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

#undef __SYCL_CONSTEXPR_ON_DEVICE
#undef _CPP14_CONSTEXPR
