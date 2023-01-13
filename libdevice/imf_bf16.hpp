//==------- imf_bf16.hpp - BFloat16 emulation for intel math functions -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __LIBDEVICE_BF16_EMUL_H__
#define __LIBDEVICE_BF16_EMUL_H__

#include "device.h"
#include "imf_impl_utils.hpp"
#include <cstdint>
#include <limits>
#include <type_traits>

// Currently, we use uint16_t to emulate BFloat16 for all device.
typedef uint16_t _iml_bf16_internal;

static inline float __bfloat162float(_iml_bf16_internal b) {
  uint16_t bf16_mant = b & 0x7F;
  uint16_t bf16_sign_exp = (b & 0xFF80);
  uint32_t f32_sign_exp = static_cast<uint32_t>(bf16_sign_exp) << 16;
  uint32_t f32_mant = static_cast<uint32_t>(bf16_mant) << 16;
  return __builtin_bit_cast(float, f32_sign_exp | f32_mant);
};

static inline _iml_bf16_internal
__float2bfloat16(float f, __iml_rounding_mode rounding_mode) {
  union {
    float f_val;
    uint32_t u32_val;
  } fp32_bits;

  fp32_bits.f_val = f;
  uint16_t bf16_sign =
      static_cast<uint16_t>((fp32_bits.u32_val & 0x80000000) >> 31);
  uint16_t bf16_exp =
      static_cast<uint16_t>((fp32_bits.u32_val & 0x7F800000) >> 23);
  uint32_t f_mant = fp32_bits.u32_val & 0x7FFFFF;
  uint16_t bf16_mant = static_cast<uint16_t>(f_mant >> 16);
  // +/-infinity and NAN
  if (bf16_exp == 0xFF) {
    if (!f_mant)
      return bf16_sign ? 0xFF80 : 0x7F80;
    else
      return (bf16_sign << 15) | (bf16_exp << 7) | bf16_mant;
  }

  // +/-0
  if (!bf16_exp && !f_mant) {
    return bf16_sign ? 0x8000 : 0x0;
  }

  uint16_t mant_discard = static_cast<uint16_t>(f_mant & 0xFFFF);
  switch (rounding_mode) {
  case __IML_RTN:
    if (bf16_sign && mant_discard)
      bf16_mant++;
    break;
  case __IML_RTZ:
    break;
  case __IML_RTP:
    if (!bf16_sign && mant_discard)
      bf16_mant++;
    break;
  case __IML_RTE:
    if ((mant_discard > 0x8000) ||
        ((mant_discard == 0x8000) && ((bf16_mant & 0x1) == 0x1)))
      bf16_mant++;
    break;
  }

  // if overflow happens, bf16_exp will be 0xFF and bf16_mant will be 0,
  // infinity will be returned.
  if (bf16_mant == 0x80) {
    bf16_mant = 0;
    bf16_exp++;
  }

  return (bf16_sign << 15) | (bf16_exp << 7) | bf16_mant;
}

template <typename Ty>
static Ty __iml_bfloat162integral_u(uint16_t b,
                                    __iml_rounding_mode rounding_mode) {
  static_assert(
      std::is_unsigned<Ty>::value && std::is_integral<Ty>::value,
      "__iml_bfloat162integral_u only accepts unsigned integral type.");
  uint16_t b_sign = b >> 15;
  // return 0 for all negative bfloat16 when converting them to unsigned
  // integral type.
  if (b_sign)
    return 0;
  uint16_t b_exp = b >> 7;
  uint16_t b_mant = b & 0x7F;
  int16_t b_exp1 = static_cast<int16_t>(b_exp) - 127;

  if (!b_exp)
    return (b_mant && (__IML_RTP == rounding_mode)) ? 1 : 0;

  // convert infinity to max.
  if (b_exp == 0xFF && !b_mant)
    return std::numeric_limits<Ty>::max();

  // According to CUDA math docs, for u/short, u/int type, return 0 for NAN
  // and return 0x80000000 for NAN when converting bfloat16 to u/ll.
  if (b_exp == 0xFF && b_mant)
    return (sizeof(Ty) < 8) ? 0 : static_cast<Ty>(0x8000000000000000ULL);

  // Normalized value can be represented as 1.signifcand * 2^b_exp1
  // and is equivalent to 1.signifcand * 2^7 * 2^(b_exp1 - 7).
  // -133 <= b_exp1 - 7 <= 120
  Ty x_val = b_mant;
  Ty x_discard;
  x_val |= (0x1 << 7);
  b_exp1 -= 7;
  if (b_exp1 >= 0 && b_exp1 <= static_cast<int16_t>(sizeof(Ty) * 8 - 8))
    return (x_val <<= b_exp1);
  if (b_exp1 > static_cast<int16_t>(sizeof(Ty) * 8 - 8))
    return std::numeric_limits<Ty>::max();

  // if b_exp1 < 0, we need to right shift and discard some bits, when
  // -b_exp1 > 8, the  value will be less than 0.5 and we don't need to
  // take special care for RTE.
  if (-b_exp1 > 8)
    return (__IML_RTP == rounding_mode) ? 1 : 0;

  x_discard = x_val & ((static_cast<Ty>(1) << -b_exp1) - 1);
  Ty mid = 1 << (-b_exp1 - 1);
  x_val >>= -b_exp1;
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
static Ty __iml_bfloat162integral_s(uint16_t b,
                                    __iml_rounding_mode rounding_mode) {
  static_assert(std::is_signed<Ty>::value && std::is_integral<Ty>::value,
                "__iml_bfloat162integral_s only accepts signed integral type.");
  typedef typename __iml_get_unsigned<Ty>::utype UTy;
  uint16_t b_sign = b >> 15;
  uint16_t b_exp = (b & static_cast<uint16_t>(0x7F80)) >> 7;
  uint16_t b_mant = b & 0x7F;
  int16_t b_exp1 = static_cast<int16_t>(b_exp) - 127;

  if (!b_exp) {
    if (!b_mant)
      return 0;
    else if (b_sign && (__IML_RTN == rounding_mode))
      return -1;
    else if (!b_sign && (__IML_RTP == rounding_mode))
      return 1;
    else
      return 0;
  }

  if (b_exp == 0xFF) {
    if (b_mant) {
      return (sizeof(Ty) < 8) ? 0 : static_cast<Ty>(0x8000000000000000ULL);
    } else
      return b_sign ? std::numeric_limits<Ty>::min()
                    : std::numeric_limits<Ty>::max();
  }

  // Normalized value can be represented as 1.signifcand * 2^b_exp1
  // and is equivalent to 1.signifcand * 2^7 * 2^(b_exp1 - 7).
  // -133 <= b_exp1 - 7 <= 120
  UTy x_val = b_mant;
  UTy x_discard;
  x_val |= (0x1 << 7);
  b_exp1 -= 7;
  // Overflow happens
  if (b_exp1 >= static_cast<int16_t>((sizeof(Ty) * 8) - 8)) {
    return b_sign ? std::numeric_limits<Ty>::min()
                  : std::numeric_limits<Ty>::max();
  }

  if (b_exp1 >= 0) {
    x_val <<= b_exp1;
    return !b_sign ? x_val : (~x_val + 1);
  }

  // b_exp1 < 0, need right shift  -b_exp1 bits, if -b_exp1 > 8, the value
  // is less than 0.5, so don't need to take special care for RTE
  if (-b_exp1 > 8) {
    if (b_sign && (__IML_RTN == rounding_mode))
      return -1;
    if (!b_sign && (__IML_RTP == rounding_mode))
      return 1;
    return 0;
  }

  x_discard = x_val & ((static_cast<UTy>(1) << -b_exp1) - 1);
  UTy mid = static_cast<UTy>(1) << (-b_exp1 - 1);
  x_val >>= -b_exp1;
  if (!x_discard)
    return x_val;
  switch (rounding_mode) {
  case __IML_RTE:
    if ((x_discard > mid) || ((x_discard == mid) && ((x_val & 0x1) == 0x1)))
      x_val++;
    break;
  case __IML_RTN:
    if (b_sign)
      x_val++;
    break;
  case __IML_RTP:
    if (!b_sign)
      x_val++;
    break;
  case __IML_RTZ:
    break;
  }

  return !b_sign ? x_val : (~x_val + 1);
}

template <typename Ty>
static uint16_t __iml_integral2bfloat16_u(Ty u,
                                          __iml_rounding_mode rounding_mode) {
  static_assert(
      std::is_unsigned<Ty>::value && std::is_integral<Ty>::value,
      "__iml_integral2bfloat16_u only accepts unsigned integral type.");
  if (!u)
    return 0;
  size_t msb_pos = get_msb_pos(u);
  // return half representation for 1
  if (msb_pos == 0)
    return 0x3F80;
  Ty mant = u & ((static_cast<Ty>(1) << msb_pos) - 1);
  // Unsigned integral value can be represented by 1.mant * (2^msb_pos),
  // msb_pos is also the bit number of mantissa, 0 < msb_pos < sizeof(Ty) * 8,
  // exponent of bfloat16 precision value range is [-126, 127].

  uint16_t b_exp = msb_pos;
  uint16_t b_mant;

  if (msb_pos <= 7) {
    mant <<= (7 - msb_pos);
    b_mant = static_cast<uint16_t>(mant);
  } else {
    b_mant = static_cast<uint16_t>(mant >> (msb_pos - 7));
    Ty mant_discard = mant & ((static_cast<Ty>(1) << (msb_pos - 7)) - 1);
    Ty mid = static_cast<Ty>(1) << (msb_pos - 8);
    switch (rounding_mode) {
    case __IML_RTE:
      if ((mant_discard > mid) ||
          ((mant_discard == mid) && ((b_mant & 0x1) == 0x1)))
        b_mant++;
      break;
    case __IML_RTP:
      if (mant_discard)
        b_mant++;
      break;
    case __IML_RTN:
    case __IML_RTZ:
      break;
    }
  }
  if (b_mant == 0x80) {
    b_exp++;
    b_mant = 0;
  }

  b_exp += 127;
  return (b_exp << 7) | b_mant;
}

template <typename Ty>
static uint16_t __iml_integral2bfloat16_s(Ty i,
                                          __iml_rounding_mode rounding_mode) {
  static_assert(std::is_signed<Ty>::value && std::is_integral<Ty>::value,
                "__iml_integral2bfloat16_s only accepts signed integral type.");
  typedef typename __iml_get_unsigned<Ty>::utype UTy;
  if (!i)
    return 0;
  uint16_t b_sign = (i >= 0) ? 0 : 0x8000;
  UTy ui = (i > 0) ? static_cast<UTy>(i) : static_cast<UTy>(-i);
  size_t msb_pos = get_msb_pos<UTy>(ui);
  if (msb_pos == 0)
    return b_sign ? 0xBF80 : 0x3F80;
  UTy mant = ui & ((static_cast<UTy>(1) << msb_pos) - 1);

  uint16_t b_exp = msb_pos;
  uint16_t b_mant;
  if (msb_pos <= 7) {
    mant <<= (7 - msb_pos);
    b_mant = static_cast<uint16_t>(mant);
  } else {
    b_mant = static_cast<uint16_t>(mant >> (msb_pos - 7));
    Ty mant_discard = mant & ((static_cast<Ty>(1) << (msb_pos - 7)) - 1);
    Ty mid = static_cast<Ty>(1) << (msb_pos - 8);
    switch (rounding_mode) {
    case __IML_RTE:
      if ((mant_discard > mid) ||
          ((mant_discard == mid) && ((b_mant & 0x1) == 0x1)))
        b_mant++;
      break;
    case __IML_RTP:
      if (mant_discard && !b_sign)
        b_mant++;
      break;
    case __IML_RTN:
      if (mant_discard && b_sign)
        b_mant++;
    case __IML_RTZ:
      break;
    }
  }

  if (b_mant == 0x80) {
    b_exp++;
    b_mant = 0;
  }
  b_exp += 127;
  return b_sign | (b_exp << 7) | b_mant;
}

// We convert bf16 to fp32 and do all arithmetic operations, then convert back.
class _iml_bf16 {
public:
  _iml_bf16(_iml_bf16_internal b) : _bf16_internal(b) {}
  _iml_bf16() = default;
  _iml_bf16(const _iml_bf16 &) = default;
  _iml_bf16 &operator=(const _iml_bf16 &rh) = default;
  _iml_bf16 &operator=(float fval) {
    _bf16_internal = __float2bfloat16(fval, __IML_RTE);
    return *this;
  }
  _iml_bf16(float fval) : _bf16_internal(__float2bfloat16(fval, __IML_RTE)) {}
  explicit operator float() const { return __bfloat162float(_bf16_internal); }

  _iml_bf16_internal get_internal() const { return _bf16_internal; }
  bool operator==(const _iml_bf16 &rh) {
    return _bf16_internal == rh._bf16_internal;
  }
  bool operator!=(const _iml_bf16 &rh) { return !operator==(rh); }

  _iml_bf16 &operator+=(const _iml_bf16 &rh) {
    *this = (operator float() + static_cast<float>(rh));
    return *this;
  }
  _iml_bf16 &operator-=(const _iml_bf16 &rh) {
    *this = (operator float() - static_cast<float>(rh));
    return *this;
  }
  _iml_bf16 &operator*=(const _iml_bf16 &rh) {
    *this = (operator float() * static_cast<float>(rh));
    return *this;
  }
  _iml_bf16 &operator/=(const _iml_bf16 &rh) {
    *this = (operator float() / static_cast<float>(rh));
    return *this;
  }
  _iml_bf16 &operator++() {
    *this = operator float() + 1.f;
    return *this;
  }
  _iml_bf16 operator++(int) {
    _iml_bf16 res(*this);
    operator++();
    return res;
  }
  _iml_bf16 &operator--() {
    *this = operator float() - 1.f;
    return *this;
  }
  _iml_bf16 operator--(int) {
    _iml_bf16 res(*this);
    operator--();
    return res;
  }

  _iml_bf16 operator-() {
    _iml_bf16 res(-operator float());
    return res;
  }

  bool operator<(const _iml_bf16 &rh) {
    return operator float() < static_cast<float>(rh);
  }
  bool operator>(const _iml_bf16 &rh) {
    return operator float() > static_cast<float>(rh);
  }

  _iml_bf16 operator+(const _iml_bf16 &rh) {
    _iml_bf16 res(*this);
    res += rh;
    return res;
  }

  _iml_bf16 operator-(const _iml_bf16 &rh) {
    _iml_bf16 res(*this);
    res -= rh;
    return res;
  }

  _iml_bf16 operator*(const _iml_bf16 &rh) {
    _iml_bf16 res(*this);
    res *= rh;
    return res;
  }

  _iml_bf16 operator/(const _iml_bf16 &rh) {
    _iml_bf16 res(*this);
    res /= rh;
    return res;
  }
  bool operator<=(const _iml_bf16 &rh) {
    return operator<(rh) || operator==(rh);
  }
  bool operator>=(const _iml_bf16 &rh) {
    return operator>(rh) || operator==(rh);
  }

private:
  _iml_bf16_internal _bf16_internal;
};
#endif
