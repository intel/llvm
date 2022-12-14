//==-------------------- imf_bf16.hpp - bfloat16 utils ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// C++ APIs for bfloat16 util functions.
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <type_traits>

extern "C" {
float __imf_bfloat162float(uint16_t);
uint16_t __imf_float2bfloat16(float);
uint16_t __imf_float2bfloat16_rd(float);
uint16_t __imf_float2bfloat16_rn(float);
uint16_t __imf_float2bfloat16_ru(float);
uint16_t __imf_float2bfloat16_rz(float);
};

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {
namespace math {

// Need to ensure that sycl bfloat16 defined in bfloat16.hpp is compatible
// with uint16_t in layout.
#if __cplusplus >= 201703L
static_assert(sizeof(sycl::ext::oneapi::bfloat16) == sizeof(uint16_t),
              "sycl bfloat16 is not compatible with uint16_t.");
// Bfloat16 type cast utils
float bfloat162float(sycl::ext::oneapi::bfloat16 b) {
  return __imf_bfloat162float(__builtin_bit_cast(uint16_t, b));
}

sycl::ext::oneapi::bfloat16 float2bfloat16(float f) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16(f));
}

sycl::ext::oneapi::bfloat16 float2bfloat16_rd(float f) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16_rd(f));
}

sycl::ext::oneapi::bfloat16 float2bfloat16_rn(float f) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16_rn(f));
}

sycl::ext::oneapi::bfloat16 float2bfloat16_ru(float f) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16_ru(f));
}

sycl::ext::oneapi::bfloat16 float2bfloat16_rz(float f) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16_rz(f));
}

sycl::float2 bfloat1622float2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
  return sycl::float2{bfloat162float(b[0]), bfloat162float(b[1])};
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
bfloat162bfloat162(sycl::ext::oneapi::bfloat16 b) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = res[1] = b;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
halves2bfloat162(sycl::ext::oneapi::bfloat16 a, sycl::ext::oneapi::bfloat16 b) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = a;
  res[1] = b;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
float22bfloat162_rn(sycl::float2 f) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = float2bfloat16_rn(f.s0());
  res[1] = float2bfloat16_rn(f.s1());
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2> float2bfloat162_rn(float f) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = res[1] = float2bfloat16_rn(f);
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2> floats2bfloat162_rn(float a,
                                                                 float b) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = float2bfloat16_rn(a);
  res[1] = float2bfloat16_rn(b);
  return res;
}

// Bfloat16 comparison utils
bool hisnan(sycl::ext::oneapi::bfloat16 b) {
  uint16_t bf16_bits = __builtin_bit_cast(uint16_t, b);
  return (((bf16_bits & 0x7F80) == 0x7F80) && (bf16_bits & 0x7F)) ? true
                                                                  : false;
}

bool hisinf(sycl::ext::oneapi::bfloat16 b) {
  uint16_t bf16_bits = __builtin_bit_cast(uint16_t, b);
  return (((bf16_bits & 0x7F80) == 0x7F80) && !(bf16_bits & 0x7F)) ? true
                                                                   : false;
}

bool heq(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  return __builtin_bit_cast(uint16_t, b1) == __builtin_bit_cast(uint16_t, b2);
}

bool hequ(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b1))
    return true;
  return __builtin_bit_cast(uint16_t, b1) == __builtin_bit_cast(uint16_t, b2);
}

bool hne(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  return __builtin_bit_cast(uint16_t, b1) != __builtin_bit_cast(uint16_t, b2);
}

bool hneu(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  return __builtin_bit_cast(uint16_t, b1) != __builtin_bit_cast(uint16_t, b2);
}

bool hge(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 >= bf2);
}

bool hgeu(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 >= bf2);
}

bool hgt(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 > bf2);
}

bool hgtu(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 > bf2);
}

bool hle(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 <= bf2);
}

bool hleu(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 <= bf2);
}

bool hlt(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 < bf2);
}

bool hltu(sycl::ext::oneapi::bfloat16 b1, sycl::ext::oneapi::bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 < bf2);
}

sycl::ext::oneapi::bfloat16 hmax(sycl::ext::oneapi::bfloat16 b1,
                                 sycl::ext::oneapi::bfloat16 b2) {
  uint16_t canonical_nan = 0x7FC0;
  uint16_t b1a = __builtin_bit_cast(uint16_t, b1);
  uint16_t b2a = __builtin_bit_cast(uint16_t, b2);
  if (hisnan(b1) && hisnan(b2))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16, canonical_nan);
  if (hisnan(b1))
    return b2;
  else if (hisnan(b2))
    return b1;
  else if (((b1a | b2a) == static_cast<uint16_t>(0x8000)) &&
           ((b1a & b2a) == 0x0))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                              static_cast<uint16_t>(0x0));
  else {
    return (hgt(b1, b2) ? b1 : b2);
  }
}

sycl::ext::oneapi::bfloat16 hmax_nan(sycl::ext::oneapi::bfloat16 b1,
                                     sycl::ext::oneapi::bfloat16 b2) {
  uint16_t canonical_nan = 0x7FC0;
  uint16_t b1a = __builtin_bit_cast(uint16_t, b1);
  uint16_t b2a = __builtin_bit_cast(uint16_t, b2);
  if (hisnan(b1) || hisnan(b2))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16, canonical_nan);
  else if (((b1a | b2a) == static_cast<uint16_t>(0x8000)) &&
           ((b1a & b2a) == 0x0))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                              static_cast<uint16_t>(0x0));
  else
    return (hgt(b1, b2) ? b1 : b2);
}

sycl::ext::oneapi::bfloat16 hmin(sycl::ext::oneapi::bfloat16 b1,
                                 sycl::ext::oneapi::bfloat16 b2) {
  uint16_t canonical_nan = 0x7FC0;
  uint16_t b1a = __builtin_bit_cast(uint16_t, b1);
  uint16_t b2a = __builtin_bit_cast(uint16_t, b2);
  if (hisnan(b1) && hisnan(b2))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16, canonical_nan);
  if (hisnan(b1))
    return b2;
  else if (hisnan(b2))
    return b1;
  else if (((b1a | b2a) == static_cast<uint16_t>(0x8000)) &&
           ((b1a & b2a) == 0x0))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                              static_cast<uint16_t>(0x8000));
  else {
    return (hlt(b1, b2) ? b1 : b2);
  }
}

sycl::ext::oneapi::bfloat16 hmin_nan(sycl::ext::oneapi::bfloat16 b1,
                                     sycl::ext::oneapi::bfloat16 b2) {
  uint16_t canonical_nan = 0x7FC0;
  uint16_t b1a = __builtin_bit_cast(uint16_t, b1);
  uint16_t b2a = __builtin_bit_cast(uint16_t, b2);
  if (hisnan(b1) || hisnan(b2))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16, canonical_nan);
  else if (((b1a | b2a) == static_cast<uint16_t>(0x8000)) &&
           ((b1a & b2a) == 0x0))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                              static_cast<uint16_t>(0x8000));
  else
    return (hlt(b1, b2) ? b1 : b2);
}

bool hbeq2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
           sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return heq(b1[0], b2[0]) && heq(b1[1], b2[1]);
}

bool hbequ2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hequ(b1[0], b2[0]) && hequ(b1[1], b2[1]);
}

bool hbge2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
           sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hge(b1[0], b2[0]) && hge(b1[1], b2[1]);
}

bool hbgeu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hgeu(b1[0], b2[0]) && hgeu(b1[1], b2[1]);
}

bool hbgt2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
           sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hgt(b1[0], b2[0]) && hgt(b1[1], b2[1]);
}

bool hbgtu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hgtu(b1[0], b2[0]) && hgtu(b1[1], b2[1]);
}

bool hble2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
           sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hle(b1[0], b2[0]) && hle(b1[1], b2[1]);
}

bool hbleu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hleu(b1[0], b2[0]) && hleu(b1[1], b2[1]);
}

bool hblt2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
           sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hlt(b1[0], b2[0]) && hlt(b1[1], b2[1]);
}

bool hbltu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hltu(b1[0], b2[0]) && hltu(b1[1], b2[1]);
}

bool hbne2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
           sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hne(b1[0], b2[0]) && hne(b1[1], b2[1]);
}

bool hbneu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
            sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return hneu(b1[0], b2[0]) && hneu(b1[1], b2[1]);
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
heq2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
     sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = heq(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = heq(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned heq2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                   sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (heq(b1[0], b2[0]))
    res |= 0xFFFF;
  if (heq(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hequ2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hequ(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hequ(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hequ2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hequ(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hequ(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hne2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
     sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hne(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hne(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hne2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                   sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hne(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hne(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hneu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hneu(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hneu(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hneu2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hneu(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hneu(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hge2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
     sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hge(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hge(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hge2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                   sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hge(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hge(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hgeu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hgeu(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hgeu(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hgeu2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hgeu(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hgeu(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hgt2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
     sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hgt(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hgt(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hgt2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                   sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hgt(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hgt(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hgtu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hgtu(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hgtu(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hgtu2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hgtu(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hgtu(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hisnan2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hisnan(b[0]) ? 1.0f : 0.f;
  res[1] = hisnan(b[1]) ? 1.0f : 0.f;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hle2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
     sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hle(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hle(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hle2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                   sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hle(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hle(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hleu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hleu(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hleu(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hleu2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hleu(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hleu(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hlt2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
     sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hlt(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hlt(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hlt2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                   sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hlt(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hlt(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hltu2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hltu(b1[0], b2[0]) ? 1.0f : 0.f;
  res[1] = hltu(b1[1], b2[0]) ? 1.0f : 0.f;
  return res;
}

unsigned hltu2_mask(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
                    sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  unsigned res = 0;
  if (hltu(b1[0], b2[0]))
    res |= 0xFFFF;
  if (hltu(b1[1], b2[1]))
    res |= 0xFFFF0000;
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hmax2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hmax(b1[0], b2[0]);
  res[1] = hmax(b1[1], b2[0]);
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hmax2_nan(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
          sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hmax_nan(b1[0], b2[0]);
  res[1] = hmax_nan(b1[1], b2[0]);
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hmin2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hmin(b1[0], b2[0]);
  res[1] = hmin(b1[1], b2[0]);
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hmin2_nan(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
          sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res;
  res[0] = hmin_nan(b1[0], b2[0]);
  res[1] = hmin_nan(b1[1], b2[0]);
  return res;
}

// Bfloat16 Arithmetic utils
sycl::ext::oneapi::bfloat16 hneg(sycl::ext::oneapi::bfloat16 b) {
  uint16_t bf16_bits = __builtin_bit_cast(uint16_t, b);
  uint16_t bf16_bits_n = bf16_bits ^ static_cast<uint16_t>(0x8000);
  return hisnan(b)
             ? b
             : (__builtin_bit_cast(sycl::ext::oneapi::bfloat16, bf16_bits_n));
}

sycl::ext::oneapi::bfloat16 habs(sycl::ext::oneapi::bfloat16 b) {
  uint16_t bf16_bits = __builtin_bit_cast(uint16_t, b);
  return (hisnan(b) || !(bf16_bits & static_cast<uint16_t>(0x8000))) ? b
                                                                     : hneg(b);
}

sycl::ext::oneapi::bfloat16 hadd(sycl::ext::oneapi::bfloat16 b1,
                                 sycl::ext::oneapi::bfloat16 b2) {
  return b1 + b2;
}

sycl::ext::oneapi::bfloat16 hadd_sat(sycl::ext::oneapi::bfloat16 b1,
                                     sycl::ext::oneapi::bfloat16 b2) {
  float f = bfloat162float(b1) + bfloat162float(b2);
  return sycl::isnan(f) ? __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                                             static_cast<uint16_t>(0x0))
                        : float2bfloat16(sycl::clamp(f, 0.f, 1.0f));
}

sycl::ext::oneapi::bfloat16 hsub(sycl::ext::oneapi::bfloat16 b1,
                                 sycl::ext::oneapi::bfloat16 b2) {
  return b1 - b2;
}

sycl::ext::oneapi::bfloat16 hsub_sat(sycl::ext::oneapi::bfloat16 b1,
                                     sycl::ext::oneapi::bfloat16 b2) {
  float f = bfloat162float(b1) - bfloat162float(b2);
  return sycl::isnan(f) ? __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                                             static_cast<uint16_t>(0x0))
                        : float2bfloat16(sycl::clamp(f, 0.f, 1.0f));
}

sycl::ext::oneapi::bfloat16 hmul(sycl::ext::oneapi::bfloat16 b1,
                                 sycl::ext::oneapi::bfloat16 b2) {
  return b1 * b2;
}

sycl::ext::oneapi::bfloat16 hmul_sat(sycl::ext::oneapi::bfloat16 b1,
                                     sycl::ext::oneapi::bfloat16 b2) {
  float f = bfloat162float(b1) * bfloat162float(b2);
  return sycl::isnan(f) ? __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                                             static_cast<uint16_t>(0x0))
                        : float2bfloat16(sycl::clamp(f, 0.f, 1.0f));
}

sycl::ext::oneapi::bfloat16 hdiv(sycl::ext::oneapi::bfloat16 b1,
                                 sycl::ext::oneapi::bfloat16 b2) {
  return b1 / b2;
}

sycl::ext::oneapi::bfloat16 hfma(sycl::ext::oneapi::bfloat16 b1,
                                 sycl::ext::oneapi::bfloat16 b2,
                                 sycl::ext::oneapi::bfloat16 b3) {
  float f1 = bfloat162float(b1);
  float f2 = bfloat162float(b2);
  float f3 = bfloat162float(b3);
  return float2bfloat16(sycl::fma(f1, f2, f3));
}

sycl::ext::oneapi::bfloat16 hfma_sat(sycl::ext::oneapi::bfloat16 b1,
                                     sycl::ext::oneapi::bfloat16 b2,
                                     sycl::ext::oneapi::bfloat16 b3) {
  float f =
      sycl::fma(bfloat162float(b1), bfloat162float(b2), bfloat162float(b3));
  return sycl::isnan(f) ? __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                                             static_cast<uint16_t>(0))
                        : float2bfloat16(sycl::clamp(f, 0.f, 1.0f));
}

sycl::ext::oneapi::bfloat16 hfma_relu(sycl::ext::oneapi::bfloat16 b1,
                                      sycl::ext::oneapi::bfloat16 b2,
                                      sycl::ext::oneapi::bfloat16 b3) {
  float f1 = bfloat162float(b1);
  float f2 = bfloat162float(b2);
  float f3 = bfloat162float(b3);
  float f4 = sycl::fma(f1, f2, f3);
  if (sycl::isnan(f4))
    return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                              static_cast<uint16_t>(0x7FC0));
  return (f4 < 0.f) ? float2bfloat16(0.f) : float2bfloat16(f4);
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
habs2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res{habs(b[0]), habs(b[1])};
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hadd2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return b1 + b2;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hadd2_sat(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
          sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res{hadd_sat(b1[0], b2[0]),
                                                   hadd_sat(b1[1], b2[1])};
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hsub2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return b1 - b2;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hsub2_sat(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
          sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res{hsub_sat(b1[0], b2[0]),
                                                   hsub_sat(b1[1], b2[1])};
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hmul2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return b1 * b2;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hmul2_sat(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
          sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res{hmul_sat(b1[0], b2[0]),
                                                   hmul_sat(b1[1], b2[1])};
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hdiv2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2) {
  return b1 / b2;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hneg2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res{hneg(b[0]), hneg(b[1])};
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hfma2(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2,
      sycl::marray<sycl::ext::oneapi::bfloat16, 2> b3) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res{hfma(b1[0], b2[0], b3[0]),
                                                   hfma(b1[1], b2[1], b3[1])};
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hfma2_sat(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
          sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2,
          sycl::marray<sycl::ext::oneapi::bfloat16, 2> b3) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res{
      hfma_sat(b1[0], b2[0], b3[0]), hfma_sat(b1[1], b2[1], b3[1])};
  return res;
}

sycl::marray<sycl::ext::oneapi::bfloat16, 2>
hfma2_relu(sycl::marray<sycl::ext::oneapi::bfloat16, 2> b1,
           sycl::marray<sycl::ext::oneapi::bfloat16, 2> b2,
           sycl::marray<sycl::ext::oneapi::bfloat16, 2> b3) {
  sycl::marray<sycl::ext::oneapi::bfloat16, 2> res{
      hfma_relu(b1[0], b2[0], b3[0]), hfma_relu(b1[1], b2[1], b3[1])};
  return res;
}

#endif
} // namespace math
} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
