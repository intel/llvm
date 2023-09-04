//==--------- imf_half.hpp - half emulation for intel math functions -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __LIBDEVICE_HALF_EMUL_H__
#define __LIBDEVICE_HALF_EMUL_H__

#include "device.h"
#include "imf_impl_utils.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#ifdef __LIBDEVICE_IMF_ENABLED__

#if defined(__SPIR__)
typedef _Float16 _iml_half_internal;
#else
typedef uint16_t _iml_half_internal;
#endif

static uint16_t __iml_half_exp_mask = 0x7C00;

static uint16_t __iml_half_overflow_handle(__iml_rounding_mode rounding_mode,
                                           uint16_t sign) {
  if (rounding_mode == __IML_RTZ) {
    return (sign << 15) | 0x7BFF;
  }

  if (rounding_mode == __IML_RTP && sign) {
    return 0xFBFF;
  }
  if (rounding_mode == __IML_RTN && !sign) {
    return 0x7BFF;
  }
  return (sign << 15) | 0x7C00;
}

static uint16_t __iml_half_underflow_handle(__iml_rounding_mode rounding_mode,
                                            uint16_t sign) {
  if (rounding_mode == __IML_RTN && sign) {
    return 0x8001;
  }

  if (rounding_mode == __IML_RTP && !sign) {
    return 0x1;
  }
  return (sign << 15);
}

template <typename Ty>
static uint16_t __iml_fp2half(Ty x, __iml_rounding_mode rounding_mode) {
  typedef typename __iml_fp_config<Ty>::utype UTy;
  typedef typename __iml_fp_config<Ty>::stype STy;
  union {
    Ty xf;
    UTy xu;
  } xs;

  // extract sign bit
  UTy one_bit = 0x1;
  xs.xf = x;
  uint16_t h_sign = xs.xu >> (sizeof(Ty) * 8 - 1);
  // extract exponent and mantissa
  UTy x_exp = (xs.xu >> (std::numeric_limits<Ty>::digits - 1)) &
              (__iml_fp_config<Ty>::exp_mask);
  UTy x_mant = xs.xu & ((one_bit << (std::numeric_limits<Ty>::digits - 1)) - 1);
  STy x_exp1 = x_exp - std::numeric_limits<Ty>::max_exponent + 1;
  uint16_t h_exp = static_cast<uint16_t>(x_exp1 + 15);
  uint16_t mant_shift = std::numeric_limits<Ty>::digits - 11;
  if (x_exp == __iml_fp_config<Ty>::exp_mask) {
    uint16_t res;
    if (x_mant) {
      // NAN.
      uint16_t h_mant = static_cast<uint16_t>(
          x_mant >> (std::numeric_limits<Ty>::digits - 11));
      h_mant |= 0x200;
      res = (h_sign << 15) | __iml_half_exp_mask | h_mant;
    } else {
      // Infinity, zero mantissa
      res = (h_sign << 15) | __iml_half_exp_mask;
    }
    return res;
  }

  if (!x_exp && !x_mant) {
    return (h_sign << 15);
  }

  // overflow happens
  if (x_exp1 > 15) {
    return __iml_half_overflow_handle(rounding_mode, h_sign);
  }

  // underflow, if x < minmum denormal half value.
  if (x_exp1 < -25) {
    return __iml_half_underflow_handle(rounding_mode, h_sign);
  }

  // some number should be encoded as denorm number when converting to half
  // minimum positive normalized half value is 2^-14
  if (x_exp1 < -14) {
    h_exp = 0;
    x_mant |= (one_bit << (std::numeric_limits<Ty>::digits - 1));
    mant_shift = -x_exp1 - 14 + std::numeric_limits<Ty>::digits - 11;
  }

  uint16_t h_mant = (uint16_t)(x_mant >> mant_shift);
  // Used to get discarded mantissa from original fp value.
  UTy mant_discard_mask = ((UTy)1 << mant_shift) - 1;
  UTy mid_val = (UTy)1 << (mant_shift - 1);
  switch (rounding_mode) {
  case __IML_RTZ:
    break;
  case __IML_RTP:
    if ((x_mant & mant_discard_mask) && !h_sign) {
      ++h_mant;
    }
    break;
  case __IML_RTN:
    if ((x_mant & mant_discard_mask) && h_sign) {
      ++h_mant;
    }
    break;
  case __IML_RTE: {
    UTy tmp = x_mant & mant_discard_mask;
    if ((tmp > mid_val) || ((tmp == mid_val) && ((h_mant & 0x1) == 0x1))) {
      ++h_mant;
    }
    break;
  }
  }

  if (h_mant & 0x400) {
    h_exp += 1;
    h_mant = 0;
  }
  return (h_sign << 15) | (h_exp << 10) | h_mant;
}

template <typename Ty>
static Ty __iml_half2integral_u(uint16_t h, __iml_rounding_mode rounding_mode) {
  static_assert(std::is_unsigned<Ty>::value && std::is_integral<Ty>::value,
                "__iml_half2integral_u only accepts unsigned integral type.");
  uint16_t h_sign = h >> 15;
  uint16_t h_exp = (h >> 10) & 0x1F;
  uint16_t h_mant = h & 0x3FF;
  int16_t h_exp1 = (int16_t)h_exp - 15;
  if (h_sign)
    return 0;

  // For subnorm values, return 1 if rounding to +infinity.
  if (!h_exp)
    return (h_mant && (__IML_RTP == rounding_mode)) ? 1 : 0;

  if (h_exp == 0x1F)
    return h_mant ? 0 : std::numeric_limits<Ty>::max();

  // Normalized value can be represented as 1.signifcand * 2^h_exp1
  // and is equivalent to 1.signifcand * 2^10 * 2^(h_exp1 - 10).
  // -24 <= h_exp1 - 10 <= 5
  Ty x_val = h_mant;
  Ty x_discard;
  x_val |= (0x1 << 10);
  h_exp1 -= 10;

  if (h_exp1 >= 0)
    return x_val <<= h_exp1;

  // h_exp1 < 0, need right shift  -h_exp1 bits, if -h_exp1 > 11, the value
  // is less than 0.5, so don't need to take special care for RTE
  if (-h_exp1 > 11)
    return (__IML_RTP == rounding_mode) ? 1 : 0;

  x_discard = x_val & (((Ty)1 << -h_exp1) - 1);
  Ty mid = 1 << (-h_exp1 - 1);
  x_val >>= -h_exp1;
  if (!x_discard)
    return x_val;
  switch (rounding_mode) {
  case __IML_RTE:
    if ((x_discard > mid) || ((x_discard == mid) && ((x_val & 0x1) == 0x1)))
      x_val++;
    break;
  case __IML_RTN:
    break;
  case __IML_RTP:
    x_val++;
    break;
  case __IML_RTZ:
    break;
  }

  return x_val;
}

template <typename Ty>
static Ty __iml_half2integral_s(uint16_t h, __iml_rounding_mode rounding_mode) {
  static_assert(std::is_signed<Ty>::value && std::is_integral<Ty>::value,
                "__iml_half2integral_s only accepts signed integral type.");
  typedef typename __iml_get_unsigned<Ty>::utype UTy;
  uint16_t h_sign = h >> 15;
  uint16_t h_exp = (h >> 10) & 0x1F;
  uint16_t h_mant = h & 0x3FF;
  int h_exp1 = (int16_t)h_exp - 15;
  if (!h_exp) {
    if (!h_mant)
      return 0;
    else {
      // For subnormal values
      if (h_sign && (__IML_RTN == rounding_mode))
        return -1;
      if (!h_sign && (__IML_RTP == rounding_mode))
        return 1;
      return 0;
    }
  }

  if (h_exp == 0x1F) {
    // For NAN, return 0
    if (h_mant) {
      return 0;
    } else {
      // For +/-infinity value, return max and min integral value
      return h_sign ? std::numeric_limits<Ty>::min()
                    : std::numeric_limits<Ty>::max();
    }
  }

  // Normalized value can be represented as 1.signifcand * 2^h_exp1
  // and is equivalent to 1.signifcand * 2^10 * 2^(h_exp1 - 10).
  // -24 <= h_exp1 - 10 <= 5
  UTy x_val = h_mant;
  UTy x_discard;
  x_val |= (0x1 << 10);
  h_exp1 -= 10;
  // Overflow happens
  if (h_exp1 >= (int)((sizeof(Ty) * 8) - 11)) {
    return h_sign ? std::numeric_limits<Ty>::min()
                  : std::numeric_limits<Ty>::max();
  }

  if (h_exp1 >= 0) {
    x_val <<= h_exp1;
    return !h_sign ? x_val : (~x_val + 1);
  }

  // h_exp1 < 0, need right shift  -h_exp1 bits, if -h_exp1 > 11, the value
  // is less than 0.5, so don't need to take special care for RTE
  if (-h_exp1 > 11) {
    if (h_sign && (__IML_RTN == rounding_mode))
      return -1;
    if (!h_sign && (__IML_RTP == rounding_mode))
      return 1;
    return 0;
  }

  x_discard = x_val & (((UTy)1 << -h_exp1) - 1);
  UTy mid = (UTy)1 << (-h_exp1 - 1);
  x_val >>= -h_exp1;
  if (!x_discard)
    return x_val;
  switch (rounding_mode) {
  case __IML_RTE:
    if ((x_discard > mid) || ((x_discard == mid) && ((x_val & 0x1) == 0x1)))
      x_val++;
    break;
  case __IML_RTN:
    if (h_sign)
      x_val++;
    break;
  case __IML_RTP:
    if (!h_sign)
      x_val++;
    break;
  case __IML_RTZ:
    break;
  }

  return !h_sign ? x_val : (~x_val + 1);
}

template <typename Ty>
static uint16_t __iml_integral2half_u(Ty u, __iml_rounding_mode rounding_mode) {
  static_assert(std::is_unsigned<Ty>::value && std::is_integral<Ty>::value,
                "__iml_integral2half_u only accepts unsigned integral type.");
  if (!u)
    return 0;
  size_t msb_pos = get_msb_pos(u);
  // return half representation for 1
  if (msb_pos == 0)
    return 0x3C00;
  Ty mant = u & (((Ty)1 << msb_pos) - 1);
  // Unsigned integral value can be represented by 1.mant * (2^msb_pos),
  // msb_pos is also the bit number of mantissa, 0 < msb_pos < sizeof(Ty) * 8,
  // exponent of half precision value range is [-14, 15].
  bool is_overflow = false;
  if (msb_pos > 15)
    is_overflow = true;

  uint16_t h_exp = msb_pos;
  uint16_t h_mant;
  if (!is_overflow) {
    if (msb_pos <= 10) {
      mant <<= (10 - msb_pos);
      h_mant = (uint16_t)mant;
    } else {
      h_mant = (uint16_t)(mant >> (msb_pos - 10));
      Ty mant_discard = mant & (((Ty)1 << (msb_pos - 10)) - 1);
      Ty mid = (Ty)1 << (msb_pos - 11);
      switch (rounding_mode) {
      case __IML_RTE:
        if ((mant_discard > mid) ||
            ((mant_discard == mid) && ((h_mant & 0x1) == 0x1)))
          h_mant++;
        break;
      case __IML_RTP:
        if (mant_discard)
          h_mant++;
        break;
      case __IML_RTN:
      case __IML_RTZ:
        break;
      }
    }
    if (h_mant == 0x400) {
      h_exp++;
      h_mant = 0;
      if (h_exp > 15)
        is_overflow = true;
    }
  }

  if (is_overflow) {
    // According to IEEE-754 standards(Ch 7.4), RTE carries all overflows
    // to infinity with sign, RTZ carries all overflows to format's largest
    // finite number with sign, RTN carries positive overflows to format's
    // largest finite number and carries negative overflows to -infinity.
    // RTP carries negative overflows to the format's most negative finite
    // number and carries positive overflow to +infinity.
    if (__IML_RTZ == rounding_mode || __IML_RTN == rounding_mode)
      return 0x7BFF;
    else
      return 0x7C00;
  }
  h_exp += 15;
  return (h_exp << 10) | h_mant;
}

template <typename Ty>
static uint16_t __iml_integral2half_s(Ty i, __iml_rounding_mode rounding_mode) {
  static_assert(std::is_signed<Ty>::value && std::is_integral<Ty>::value,
                "__iml_integral2half_s only accepts unsigned integral type.");

  typedef typename __iml_get_unsigned<Ty>::utype UTy;
  if (!i)
    return 0;
  uint16_t h_sign = (i >= 0) ? 0 : 0x8000;
  UTy ui = (i > 0) ? static_cast<UTy>(i) : static_cast<UTy>(-i);
  size_t msb_pos = get_msb_pos<UTy>(ui);
  if (msb_pos == 0)
    return h_sign ? 0xBC00 : 0x3C00;
  UTy mant = ui & (((UTy)1 << msb_pos) - 1);
  bool is_overflow = false;
  if (msb_pos > 15)
    is_overflow = true;

  uint16_t h_exp = msb_pos;
  uint16_t h_mant;
  if (!is_overflow) {
    if (msb_pos <= 10) {
      mant <<= (10 - msb_pos);
      h_mant = (uint16_t)mant;
    } else {
      h_mant = (uint16_t)(mant >> (msb_pos - 10));
      Ty mant_discard = mant & ((1 << (msb_pos - 10)) - 1);
      Ty mid = 1 << (msb_pos - 11);
      switch (rounding_mode) {
      case __IML_RTE:
        if ((mant_discard > mid) ||
            ((mant_discard == mid) && ((h_mant & 0x1) == 0x1)))
          h_mant++;
        break;
      case __IML_RTP:
        if (mant_discard && !h_sign)
          h_mant++;
        break;
      case __IML_RTN:
        if (mant_discard && h_sign)
          h_mant++;
      case __IML_RTZ:
        break;
      }
    }
    if (h_mant == 0x400) {
      h_exp++;
      h_mant = 0;
      if (h_exp > 15)
        is_overflow = true;
    }
  }

  if (is_overflow) {
    // According to IEEE-754 standards(Ch 7.4), RTE carries all overflows
    // to infinity with sign, RTZ carries all overflows to format's largest
    // finite number with sign, RTN carries positive overflows to format's
    // largest finite number and carries negative overflows to -infinity.
    // RTP carries negative overflows to the format's most negative finite
    // number and carries positive overflow to +infinity.
    if (__IML_RTE == rounding_mode || ((__IML_RTP == rounding_mode) && !h_sign))
      return h_sign ? 0xFC00 : 0x7C00;
    if (__IML_RTZ == rounding_mode ||
        ((__IML_RTN == rounding_mode) && !h_sign) ||
        ((__IML_RTP == rounding_mode) && h_sign))
      return h_sign ? 0xFBFF : 0x7BFF;
    return 0xFC00;
  }
  h_exp += 15;
  return h_sign | (h_exp << 10) | h_mant;
}

static inline _iml_half_internal __float2half(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __iml_fp2half<float>(x, __IML_RTE);
#elif defined(__SPIR__)
  return __spirv_FConvert_Rhalf_rte(x);
#endif
}

static inline float __half2float(_iml_half_internal x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  // Extract the sign from the bits. It is 1 if the sign is negative
  const uint32_t sign = static_cast<uint32_t>(x & 0x8000) << 16;
  // Extract the exponent from the bits
  const uint8_t exp16 = (x & 0x7c00) >> 10;
  // Extract the fraction from the bits
  uint16_t frac16 = x & 0x3ff;

  uint32_t exp32 = 0;
  if (__builtin_expect(exp16 == 0x1f, 0)) {
    exp32 = 0xff;
  } else if (__builtin_expect(exp16 == 0, 0)) {
    exp32 = 0;
  } else {
    exp32 = static_cast<uint32_t>(exp16) + 112;
  }
  // corner case: subnormal -> normal
  // The denormal number of FP16 can be represented by FP32, therefore we need
  // to recover the exponent and recalculate the fration.
  if (__builtin_expect(exp16 == 0 && frac16 != 0, 0)) {
    uint8_t offset = 0;
    do {
      ++offset;
      frac16 <<= 1;
    } while ((frac16 & 0x400) != 0x400);
    // mask the 9th bit
    frac16 &= 0x3ff;
    exp32 = 113 - offset;
  }

  uint32_t frac32 = frac16 << 13;

  uint32_t fp32_bits = 0;
  fp32_bits |= sign;
  fp32_bits |= (exp32 << 23);
  fp32_bits |= frac32;
  return __builtin_bit_cast(float, fp32_bits);
#elif defined(__SPIR__)
  return __spirv_FConvert_Rfloat_rte(x);
#endif
}

class _iml_half {
public:
  _iml_half(_iml_half_internal h) : _half_internal(h) {}
  _iml_half() = default;
  _iml_half(const _iml_half &) = default;
  _iml_half &operator=(const _iml_half &rh) = default;
  _iml_half &operator=(float fval) {
    _half_internal = __float2half(fval);
    return *this;
  }
  _iml_half(float fval) : _half_internal(__float2half(fval)) {}
  explicit operator float() const { return __half2float(_half_internal); }

  _iml_half_internal get_internal() const { return _half_internal; }
  bool operator==(const _iml_half &rh) {
    return _half_internal == rh._half_internal;
  }
  bool operator!=(const _iml_half &rh) { return !operator==(rh); }
#if (__SPIR__)
  _iml_half &operator+=(const _iml_half &rh) {
    _half_internal += rh._half_internal;
    return *this;
  }
  _iml_half &operator-=(const _iml_half &rh) {
    _half_internal -= rh._half_internal;
    return *this;
  }
  _iml_half &operator*=(const _iml_half &rh) {
    _half_internal *= rh._half_internal;
    return *this;
  }
  _iml_half &operator/=(const _iml_half &rh) {
    _half_internal /= rh._half_internal;
    return *this;
  }
  _iml_half &operator++() {
    _half_internal += 1;
    return *this;
  }
  _iml_half operator++(int) {
    _iml_half res(*this);
    operator++();
    return res;
  }
  _iml_half &operator--() {
    _half_internal -= 1;
    return *this;
  }
  _iml_half operator--(int) {
    _iml_half res(*this);
    operator--();
    return res;
  }

  _iml_half operator-() {
    _iml_half res(-_half_internal);
    return res;
  }

  bool operator<(const _iml_half &rh) {
    return _half_internal < rh._half_internal;
  }
  bool operator>(const _iml_half &rh) {
    return _half_internal > rh._half_internal;
  }
#else
  _iml_half &operator+=(const _iml_half &rh) {
    *this = (operator float() + static_cast<float>(rh));
    return *this;
  }
  _iml_half &operator-=(const _iml_half &rh) {
    *this = (operator float() - static_cast<float>(rh));
    return *this;
  }
  _iml_half &operator*=(const _iml_half &rh) {
    *this = (operator float() * static_cast<float>(rh));
    return *this;
  }
  _iml_half &operator/=(const _iml_half &rh) {
    *this = (operator float() / static_cast<float>(rh));
    return *this;
  }
  _iml_half &operator++() {
    *this = operator float() + 1;
    return *this;
  }
  _iml_half operator++(int) {
    _iml_half res(*this);
    operator++();
    return res;
  }
  _iml_half &operator--() {
    *this = operator float() - 1;
    return *this;
  }
  _iml_half operator--(int) {
    _iml_half res(*this);
    operator--();
    return res;
  }

  _iml_half operator-() {
    _iml_half res(-operator float());
    return res;
  }

  bool operator<(const _iml_half &rh) {
    return operator float() < static_cast<float>(rh);
  }
  bool operator>(const _iml_half &rh) {
    return operator float() > static_cast<float>(rh);
  }
#endif
  _iml_half operator+(const _iml_half &rh) {
    _iml_half res(*this);
    res += rh;
    return res;
  }

  _iml_half operator-(const _iml_half &rh) {
    _iml_half res(*this);
    res -= rh;
    return res;
  }

  _iml_half operator*(const _iml_half &rh) {
    _iml_half res(*this);
    res *= rh;
    return res;
  }

  _iml_half operator/(const _iml_half &rh) {
    _iml_half res(*this);
    res /= rh;
    return res;
  }
  bool operator<=(const _iml_half &rh) {
    return operator<(rh) || operator==(rh);
  }
  bool operator>=(const _iml_half &rh) {
    return operator>(rh) || operator==(rh);
  }

private:
  _iml_half_internal _half_internal;
};

#endif // __LIBDEVICE_IMF_ENABLED__
#endif // __LIBDEVICE_HALF_EMUL_H__
