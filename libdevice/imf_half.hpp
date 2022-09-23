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
#include <limits>

#ifdef __LIBDEVICE_IMF_ENABLED__

#if defined(__SPIR__)
typedef _Float16 _iml_half_internal;
#else
typedef uint16_t _iml_half_internal;
#endif

// We don't want to include fenv.h and rounding mode are used internally
// by type convert functions, so we define ourselves'.
typedef enum {
  __IML_RTE, // round to nearest-even
  __IML_RTZ, // round to zero
  __IML_RTP, // round to +inf
  __IML_RTN, // round to -inf
} __iml_rounding_mode;

static uint16_t __iml_half_exp_mask = 0x7C00;
template <typename Ty> struct __iml_select_int {};

template <> struct __iml_select_int<float> {
  using utype = uint32_t;
  using stype = int32_t;
  const static uint32_t exp_mask = 0xFF;
};

template <> struct __iml_select_int<double> {
  using utype = uint64_t;
  using stype = int64_t;
  const static uint64_t exp_mask = 0x7FF;
};

static uint16_t __iml_half_overflow_handle(int rounding_mode, uint16_t sign) {
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

static uint16_t __iml_half_underflow_handle(int rounding_mode, uint16_t sign) {
  if (rounding_mode == __IML_RTN && sign) {
    return 0x8001;
  }

  if (rounding_mode == __IML_RTP && !sign) {
    return 0x1;
  }
  return (sign << 15);
}

template <typename Ty>
static uint16_t __iml_half2fp(Ty x, int rounding_mode) {
  typedef typename __iml_select_int<Ty>::utype TYU;
  typedef typename __iml_select_int<Ty>::stype TYS;
  union {
    Ty xf;
    TYU xu;
  } xs;

  // extract sign bit
  TYU one_bit = 0x1;
  xs.xf = x;
  uint16_t h_sign = xs.xu >> (sizeof(Ty) * 8 - 1);
  // extract exponent and mantissa
  TYU x_exp = (xs.xu >> (std::numeric_limits<Ty>::digits - 1)) &
              (__iml_select_int<Ty>::exp_mask);
  TYU x_mant = xs.xu & ((one_bit << (std::numeric_limits<Ty>::digits - 1)) - 1);
  TYS x_exp1 = x_exp - std::numeric_limits<Ty>::max_exponent + 1;
  uint16_t h_exp = static_cast<uint16_t>(x_exp1 + 15);
  uint16_t mant_shift = std::numeric_limits<Ty>::digits - 11;
  if (x_exp == __iml_select_int<Ty>::exp_mask) {
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
  TYU mant_discard_mask = (1 << mant_shift) - 1;
  TYU mid_val = 1 << (mant_shift - 1);
  switch (rounding_mode) {
    case __IML_RTZ:
      break;
    case __IML_RTP:
      if ((x_mant & mant_discard_mask) && !h_sign) { ++h_mant; }
      break;
    case __IML_RTN:
      if ((x_mant & mant_discard_mask) && h_sign) { ++h_mant; }
      break;
    case __IML_RTE: {
      TYU tmp = x_mant & mant_discard_mask;
      if ((tmp > mid_val) || ((tmp == mid_val) && ((h_mant & 0x1) == 0x1))) { ++h_mant;}
      break;
    }
  }

  if (h_mant & 0x400) {
    h_exp += 1;
    h_mant = 0;
  }
  return (h_sign << 15) | (h_exp << 10) | h_mant;
}

// TODO: need to support float to half conversion with different
// rounding mode.
static inline _iml_half_internal __float2half(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __iml_half2fp<float>(x, __IML_RTE);
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
