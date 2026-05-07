//==-------------- hf8.hpp - DPC++ Explicit SIMD API ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of SIMD hf8 type. This type represents floating point with
// 1 bit sign, 4 bit exponent, 3 bits mantissa.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/half_type.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

class hf8 {
  using storage_t = uint8_t;
  storage_t value;

public:
  hf8() = default;
  hf8(const hf8 &) = default;
  ~hf8() = default;

  // Explicit conversion functions
  // The function performs 2 consequent conversions:
  // float->half->hf8
  // Existing sycl implementation is used to convert
  // from float to half. To convert from half to hf8
  // cmc implementation is used.
  static storage_t from_float(const float &a) {
    uint16_t bits = ::sycl::detail::float2Half(a);

    constexpr uint16_t half_exponent_mask = 0x7c00;
    constexpr uint16_t half_mantissa_mask = 0x03ff;
    constexpr uint16_t half_sign_mask = 0x8000;

    constexpr unsigned half_exp_shift = 10;
    constexpr uint8_t half_bias = 15;
    constexpr uint8_t half_inf_nan_exp = 0x1f;

    constexpr uint8_t hfloat8_bias = 7;
    constexpr uint8_t hfloat8_exp_shift = 3;
    constexpr uint8_t hfloat8_inf_nan_exp = 0x0f;
    constexpr int8_t hfloat8_min_exp = -6;
    constexpr int8_t hfloat8_min_exp_denorm = -9;
    constexpr uint8_t hfloat8_implicit_one = 0x08;
    constexpr uint8_t hfloat8_inf_nan_mantissa = 0x07;

    constexpr int half_mantissa_shift = 7;
    constexpr uint16_t hfloat8_next_unsaved_mask = 1
                                                   << (half_mantissa_shift - 1);
    constexpr uint16_t hfloat8_rest_unsaved_mask =
        hfloat8_next_unsaved_mask - 1;

    const uint8_t sign = (bits & half_sign_mask) >> 8;
    const uint16_t mantissa16 = bits & half_mantissa_mask;
    const uint8_t biased_exp16 = (bits & half_exponent_mask) >> half_exp_shift;

    uint8_t biased_exp8 = 0;
    uint8_t mantissa8 = 0;

    if (biased_exp16 == half_inf_nan_exp) { // NaN or Inf
      biased_exp8 = hfloat8_inf_nan_exp;
      mantissa8 = hfloat8_inf_nan_mantissa;
    } else if (biased_exp16 != 0) {
      const int8_t exp = biased_exp16 - half_bias;

      if (exp < hfloat8_min_exp_denorm - 1) {
        // too small, flush to zero
        mantissa8 = 0;
      } else if (exp < hfloat8_min_exp) {
        // small half numbers map to hfloat8 denormals
        const auto denormal_shift = hfloat8_min_exp - exp + half_mantissa_shift;
        const auto next_bit_mask = 1 << (denormal_shift - 1);
        const auto rest_bits_mask = next_bit_mask - 1;

        auto mantissa16_with_implicit_one =
            mantissa16 | hfloat8_implicit_one << half_mantissa_shift;

        mantissa8 = mantissa16_with_implicit_one >> denormal_shift;

        const bool next = (mantissa16_with_implicit_one & next_bit_mask) != 0;
        const bool rest = (mantissa16_with_implicit_one & rest_bits_mask) != 0;
        const bool is_odd = (mantissa8 & 1) != 0;

        mantissa8 += next && (is_odd || rest);
        biased_exp8 = 0;
      } else if (exp <= 8) {
        biased_exp8 = exp + hfloat8_bias;
        mantissa8 = mantissa16 >> half_mantissa_shift;

        // Round to nearest or even
        const bool next = (mantissa16 & hfloat8_next_unsaved_mask) != 0;
        const bool rest = (mantissa16 & hfloat8_rest_unsaved_mask) != 0;
        const bool is_odd = (mantissa8 & 1) != 0;

        if (next && (is_odd || rest)) {
          if (mantissa8 != hfloat8_inf_nan_mantissa) {
            mantissa8 += 1;
          } else if (exp != 8) {
            mantissa8 = 0;
            biased_exp8++;
          }
        }
      } else {
        // large half maps to inf nan
        biased_exp8 = hfloat8_inf_nan_exp;
        mantissa8 = hfloat8_inf_nan_mantissa;
      }
    }

    return sign + (biased_exp8 << hfloat8_exp_shift) + mantissa8;
  }

  // The function performs 2 consequent conversions:
  // hf8->half->float
  // Existing sycl implementation is used to convert
  // from half to float. To convert from hf8 to half
  // cmc implementation is used.
  static float to_float(const storage_t &a) {
    constexpr uint16_t float8_exp_shift = (10 - 3);
    // -9 is needed because later we do multiplication of floats
    // 15 - 7 - 2 = 6, float exponent: 6 - 15 = -9
    // smallest hfloat8 denorm = 2^-6 * 2^-3 = 2^-9
    // 2^-9 * float(mantissa = 0x0001) = 2^-9 * 1.0
    constexpr uint16_t float8_to_16_bias_diff_denorm = ((15 - 7 - 2) << 10);
    constexpr uint16_t float8_to_16_bias_diff = ((15 - 7) << 3);
    constexpr uint16_t half_exponent_mask = 0x7c00;
    constexpr uint16_t half_mantissa_mask = 0x03ff;
    constexpr uint16_t half_sign_mask = 0x8000;
    constexpr storage_t exponent_mask = 0x78;
    constexpr storage_t mantissa_mask = 0x07;
    constexpr storage_t sign_mask = 0x80;

    const uint16_t exp = a & exponent_mask;
    const uint16_t mantissa = a & mantissa_mask;
    const uint16_t sign = (a & sign_mask) << 8;
    const uint16_t nan = (a == exponent_mask ? half_exponent_mask : 0U);
    // 0.0
    if ((exp | mantissa) == 0) {
      return ::sycl::detail::half2Float(sign);
    }
    // nan
    if (exp == exponent_mask && mantissa == mantissa_mask) {
      return ::sycl::detail::half2Float(sign | half_exponent_mask |
                                        half_mantissa_mask);
    }
    // normals
    if (exp != 0) {
      uint16_t tmp =
          (((exp + float8_to_16_bias_diff) | mantissa) << float8_exp_shift) |
          sign;
      return ::sycl::detail::half2Float(tmp);
    }
    // subnormals
    float tmpf = ::sycl::detail::half2Float(float8_to_16_bias_diff_denorm) *
                 static_cast<float>(mantissa);
    return sign ? -tmpf : tmpf;
  }

  // Implicit conversion from float to hf8
  hf8(const float &a) { value = from_float(a); }

  hf8 &operator=(const float &rhs) {
    value = from_float(rhs);
    return *this;
  }

  // Implicit conversion from hf8 to float
  operator float() const { return to_float(value); }

  // Get raw bits representation of hf8
  storage_t raw() const { return value; }

  // Logical operators (!,||,&&) are covered if we can cast to bool
  explicit operator bool() {
    constexpr storage_t sign_mask = 0x80;
    return (value & ~sign_mask);
  }

  // Unary minus operator overloading
  friend hf8 operator-(hf8 &lhs) {
    constexpr storage_t sign_mask = 0x80;
    hf8 Result(lhs);
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
