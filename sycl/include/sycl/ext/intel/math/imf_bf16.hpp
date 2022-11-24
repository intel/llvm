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
#include <sycl/ext/oneapi/experimental/bfloat16.hpp>
#include <type_traits>

using sycl_bfloat16 = sycl::ext::oneapi::experimental::bfloat16;

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
static_assert(sizeof(sycl_bfloat16) == sizeof(uint16_t),
              "sycl bfloat16 is not compatible with uint16_t.");

float bfloat162float(sycl_bfloat16 b) {
  return __imf_bfloat162float(__builtin_bit_cast(uint16_t, b));
}

sycl_bfloat16 float2bfloat16(float f) {
  return __builtin_bit_cast(sycl_bfloat16, __imf_float2bfloat16(f));
}

sycl_bfloat16 float2bfloat16_rd(float f) {
  return __builtin_bit_cast(sycl_bfloat16, __imf_float2bfloat16_rd(f));
}

sycl_bfloat16 float2bfloat16_rn(float f) {
  return __builtin_bit_cast(sycl_bfloat16, __imf_float2bfloat16_rn(f));
}

sycl_bfloat16 float2bfloat16_ru(float f) {
  return __builtin_bit_cast(sycl_bfloat16, __imf_float2bfloat16_ru(f));
}

sycl_bfloat16 float2bfloat16_rz(float f) {
  return __builtin_bit_cast(sycl_bfloat16, __imf_float2bfloat16_rz(f));
}

bool hisnan(sycl_bfloat16 b) { return sycl::isnan(bfloat162float(b)); }

bool hisinf(sycl_bfloat16 b) { return sycl::isinf(bfloat162float(b)); }

bool heq(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  return __builtin_bit_cast(uint16_t, b1) ==
         __builtin_bit_cast(uint16_t, b2);
}

bool hequ(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b1))
    return true;
  return __builtin_bit_cast(uint16_t, b1) ==
         __builtin_bit_cast(uint16_t, b2);
}

bool hne(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  return __builtin_bit_cast(uint16_t, b1) !=
         __builtin_bit_cast(uint16_t, b2);
}

bool hneu(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  return __builtin_bit_cast(uint16_t, b1) !=
         __builtin_bit_cast(uint16_t, b2);
}

bool hge(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 >= bf2);
}

bool hgeu(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 >= bf2);
}

bool hgt(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 > bf2);
}

bool hgtu(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 > bf2);
}

bool hle(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 <= bf2);
}

bool hleu(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 <= bf2);
}

bool hlt(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 < bf2);
}

bool hltu(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  float bf1 = bfloat162float(b1);
  float bf2 = bfloat162float(b2);
  return (bf1 < bf2);
}

sycl_bfloat16 hmax(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  uint16_t canonical_nan = 0x7FC0;
  uint16_t b1a = __builtin_bit_cast(uint16_t, b1);
  uint16_t b2a = __builtin_bit_cast(uint16_t, b2);
  if (hisnan(b1) && hisnan(b2))
    return __builtin_bit_cast(sycl_bfloat16, canonical_nan);
  if (hisnan(b1))
    return b2;
  else if (hisnan(b2))
    return b1;
  else if (((b1a | b2a) == 0x8000) && ((b1a & b2a) == 0x0))
    return __builtin_bit_cast(sycl_bfloat16, static_cast<uint16_t>(0x0));
  else {
    return (hgt(b1, b2) ? b1 : b2);
  }
}

sycl_bfloat16 hmax_nan(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  uint16_t canonical_nan = 0x7FC0;
  uint16_t b1a = __builtin_bit_cast(uint16_t, b1);
  uint16_t b2a = __builtin_bit_cast(uint16_t, b2);
  if (hisnan(b1) || hisnan(b2))
    return __builtin_bit_cast(sycl_bfloat16, canonical_nan);
  else if (((b1a | b2a) == 0x8000) && ((b1a & b2a) == 0x0))
    return __builtin_bit_cast(sycl_bfloat16, static_cast<uint16_t>(0x0));
  else
    return (hgt(b1, b2) ? b1 : b2);
}

sycl_bfloat16 hmin(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  uint16_t canonical_nan = 0x7FC0;
  uint16_t b1a = __builtin_bit_cast(uint16_t, b1);
  uint16_t b2a = __builtin_bit_cast(uint16_t, b2);
  if (hisnan(b1) && hisnan(b2))
    return __builtin_bit_cast(sycl_bfloat16, canonical_nan);
  if (hisnan(b1))
    return b2;
  else if (hisnan(b2))
    return b1;
  else if (((b1a | b2a) == 0x8000) && ((b1a & b2a) == 0x0))
    return __builtin_bit_cast(sycl_bfloat16, static_cast<uint16_t>(0x8000));
  else {
    return (hlt(b1, b2) ? b1 : b2);
  }
}

sycl_bfloat16 hmin_nan(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  uint16_t canonical_nan = 0x7FC0;
  uint16_t b1a = __builtin_bit_cast(uint16_t, b1);
  uint16_t b2a = __builtin_bit_cast(uint16_t, b2);
  if (hisnan(b1) || hisnan(b2))
    return __builtin_bit_cast(sycl_bfloat16, canonical_nan);
  else if (((b1a | b2a) == 0x8000) && ((b1a & b2a) == 0x0))
    return __builtin_bit_cast(sycl_bfloat16, static_cast<uint16_t>(0x8000));
  else
    return (hlt(b1, b2) ? b1 : b2);
}

#endif
} // namespace math
} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
