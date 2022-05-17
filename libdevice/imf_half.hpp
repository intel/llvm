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
#include <cstdint>

#ifdef __LIBDEVICE_IMF_ENABLED__

#if defined(__SPIR__)
typedef _Float16 _iml_half_internal;
#else
typedef uint16_t _iml_half_internal;
#endif

// TODO: need to support float to half conversion with different
// rounding mode.
static inline _iml_half_internal __float2half(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  uint32_t fp32_bits = __builtin_bit_cast(uint32_t, x);

  const uint16_t sign = (fp32_bits & 0x80000000) >> 16;
  const uint32_t frac32 = fp32_bits & 0x7fffff;
  const uint8_t exp32 = (fp32_bits & 0x7f800000) >> 23;
  const int16_t exp32_diff = exp32 - 127;

  // initialize to 0, covers the case for 0 and small numbers
  uint16_t exp16 = 0, frac16 = 0;

  if (__builtin_expect(exp32_diff > 15, 0)) {
    // Infinity and big numbers convert to infinity
    exp16 = 0x1f;
  } else if (__builtin_expect(exp32_diff > -14, 0)) {
    // normal range for half type
    exp16 = exp32_diff + 15;
    // convert 23-bit mantissa to 10-bit mantissa.
    frac16 = frac32 >> 13;
    if (frac32 >> 12 & 0x01)
      frac16 += 1;
  } else if (__builtin_expect(exp32_diff > -24, 0)) {
    // subnormals
    frac16 = (frac32 | (uint32_t(1) << 23)) >> (-exp32_diff - 1);
  }

  if (__builtin_expect(exp32 == 0xff && frac32 != 0, 0)) {
    // corner case: FP32 is NaN
    exp16 = 0x1F;
    frac16 = 0x200;
  }

  // Compose the final FP16 binary
  uint16_t res = 0;
  res |= sign;
  res |= exp16 << 10;
  res += frac16; // Add the carry bit from operation Frac16 += 1;

  return res;
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
