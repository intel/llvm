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
#include <cstdint>

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
