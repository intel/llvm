//==-------------- bf8.hpp - DPC++ Explicit SIMD API ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of SIMD bf8 type. This type represents floating point with
// 1 bit sign, 5 bit exponent, 2 bits mantissa.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops.hpp>
#include <sycl/bit_cast.hpp>
#include <sycl/half_type.hpp>

namespace sycl {
inline namespace _V1 {
  namespace ext {
  namespace intel {
  namespace experimental {
  namespace esimd {

  class bf8 {
    using storage_t = uint8_t;
    storage_t value;

  public:
    bf8() = default;
    bf8(const bf8 &) = default;
    ~bf8() = default;

    // Explicit conversion functions
    // The function performs 2 consequent conversions:
    // float->half->bf8
    // Existing sycl implementation is used to convert
    // from float to half. To convert from half to bf8
    // cmc implementation is used.
    static storage_t from_float(const float &a) {
      uint16_t tmp = ::sycl::detail::float2Half(a);
      constexpr uint16_t exponent_mask = 0x7c;
      constexpr uint16_t mantissa_mask = 0x03;
      constexpr uint16_t sign_mask = 0x80;
      constexpr uint16_t remainder_mask = 0x00ff;
      const uint8_t remainder = tmp & remainder_mask;
      constexpr uint8_t highest_representable = 0x7b;
      constexpr uint8_t bfloat8_max = 0xff;
      tmp = tmp >> 8; // Fit into 8 bits

      const bool is_nan_or_inf = (tmp & exponent_mask) == exponent_mask;

      if (!is_nan_or_inf) {
        constexpr int highest_remainder_bit = 0x80;
        if ((remainder > highest_remainder_bit && tmp != bfloat8_max) ||
            (remainder == highest_remainder_bit && tmp != bfloat8_max &&
             tmp % 2 == 1)) { // Round to nearest or even
          ++tmp;
        }
      } else {
        if (remainder != 0) { // Make sure NaN is not lost by cropping
          tmp |= mantissa_mask;
        }
      }

      return tmp;
    }

    // The function performs 2 consequent conversions:
    // bf8->half->float
    // Existing sycl implementation is used to convert
    // from half to float. To convert from bf8 to half
    // cmc implementation is used.
    static float to_float(const storage_t &a) {
      return ::sycl::detail::half2Float(a << 8);
    }

    // Implicit conversion from float to bf8
    bf8(const float &a) { value = from_float(a); }

    bf8 &operator=(const float &rhs) {
      value = from_float(rhs);
      return *this;
    }

    // Implicit conversion from bf8 to float
    operator float() const { return to_float(value); }

    // Get raw bits representation of hf8
    storage_t raw() const { return value; }

    // Logical operators (!,||,&&) are covered if we can cast to bool
    explicit operator bool() {
      constexpr storage_t sign_mask = 0x80;
      return (value & ~sign_mask);
    }

    // Unary minus operator overloading
    friend bf8 operator-(bf8 &lhs) {
      constexpr storage_t sign_mask = 0x80;
      bf8 Result(lhs);
      Result.value ^= sign_mask;
      return Result;
    }
  };

  } // namespace esimd
  } // namespace experimental
  } // namespace intel
  } // namespace ext

} // namespace _V1
} // namespace sycl

