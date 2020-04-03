//==-------------- half_type.cpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/half_type.hpp>
// This is included to enable __builtin_expect()
#include <detail/platform_util.hpp>

#include <cstring>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

static uint16_t float2Half(const float &Val) {
  const uint32_t Bits = *reinterpret_cast<const uint32_t *>(&Val);

  // Extract the sign from the float value
  const uint16_t Sign = (Bits & 0x80000000) >> 16;
  // Extract the fraction from the float value
  const uint32_t Frac32 = Bits & 0x7fffff;
  // Extract the exponent from the float value
  const uint8_t Exp32 = (Bits & 0x7f800000) >> 23;
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

static float half2Float(const uint16_t &Val) {
  // Extract the sign from the bits
  const uint32_t Sign = static_cast<uint32_t>(Val & 0x8000) << 16;
  // Extract the exponent from the bits
  const uint8_t Exp16 = (Val & 0x7c00) >> 10;
  // Extract the fraction from the bits
  uint16_t Frac16 = Val & 0x3ff;

  uint32_t Exp32 = 0;
  if (__builtin_expect(Exp16 == 0x1f, 0)) {
    Exp32 = 0xff;
  } else if (__builtin_expect(Exp16 == 0, 0)) {
    Exp32 = 0;
  } else {
    Exp32 = static_cast<uint32_t>(Exp16) + 112;
  }

  // corner case: subnormal -> normal
  // The denormal number of FP16 can be represented by FP32, therefore we need
  // to recover the exponent and recalculate the fration.
  if (__builtin_expect(Exp16 == 0 && Frac16 != 0, 0)) {
    uint8_t OffSet = 0;
    do {
      ++OffSet;
      Frac16 <<= 1;
    } while ((Frac16 & 0x400) != 0x400);
    // mask the 9th bit
    Frac16 &= 0x3ff;
    Exp32 = 113 - OffSet;
  }

  uint32_t Frac32 = Frac16 << 13;

  // Compose the final FP32 binary
  uint32_t Bits = 0;

  Bits |= Sign;
  Bits |= (Exp32 << 23);
  Bits |= Frac32;

  float Result;
  std::memcpy(&Result, &Bits, sizeof(Result));
  return Result;
}

namespace host_half_impl {

half::half(const float &RHS) : Buf(float2Half(RHS)) {}

half &half::operator+=(const half &RHS) {
  *this = operator float() + static_cast<float>(RHS);
  return *this;
}

half &half::operator-=(const half &RHS) {
  *this = operator float() - static_cast<float>(RHS);
  return *this;
}

half &half::operator*=(const half &RHS) {
  *this = operator float() * static_cast<float>(RHS);
  return *this;
}

half &half::operator/=(const half &RHS) {
  *this = operator float() / static_cast<float>(RHS);
  return *this;
}

half::operator float() const { return half2Float(Buf); }

// Operator +, -, *, /
half operator+(half LHS, const half &RHS) {
  LHS += RHS;
  return LHS;
}

half operator-(half LHS, const half &RHS) {
  LHS -= RHS;
  return LHS;
}

half operator*(half LHS, const half &RHS) {
  LHS *= RHS;
  return LHS;
}

half operator/(half LHS, const half &RHS) {
  LHS /= RHS;
  return LHS;
}

// Operator <, >, <=, >=
bool operator<(const half &LHS, const half &RHS) {
  return static_cast<float>(LHS) < static_cast<float>(RHS);
}

bool operator>(const half &LHS, const half &RHS) { return RHS < LHS; }

bool operator<=(const half &LHS, const half &RHS) { return !(LHS > RHS); }

bool operator>=(const half &LHS, const half &RHS) { return !(LHS < RHS); }

// Operator ==, !=
bool operator==(const half &LHS, const half &RHS) {
  return static_cast<float>(LHS) == static_cast<float>(RHS);
}

bool operator!=(const half &LHS, const half &RHS) { return !(LHS == RHS); }
} // namespace host_half_impl

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
