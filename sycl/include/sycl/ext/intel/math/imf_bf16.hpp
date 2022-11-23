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
using _iml_bfloat16_internal = uint16_t;

extern "C" {
float __imf_bfloat162float(_iml_bfloat16_internal);
_iml_bfloat16_internal __imf_float2bfloat16(float);
_iml_bfloat16_internal __imf_float2bfloat16_rd(float);
_iml_bfloat16_internal __imf_float2bfloat16_rn(float);
_iml_bfloat16_internal __imf_float2bfloat16_ru(float);
_iml_bfloat16_internal __imf_float2bfloat16_rz(float);
};

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {
namespace math {

// Need to ensure that sycl bfloat16 defined in bfloat16.hpp is compatible
// with uint16_t in layout.
#if __cplusplus >= 201703L
static_assert(sizeof(sycl_bfloat16) == sizeof(_iml_bfloat16_internal),
              "sycl bfloat16 is not compatible with _iml_bfloat16_internal.");

float bfloat162float(sycl_bfloat16 b) {
  return __imf_bfloat162float(__builtin_bit_cast(_iml_bfloat16_internal, b));
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

bool hisnan(sycl_bfloat b) { return sycl::isnan(bfloat162float(b)); }

bool hisinf(sycl_bfloat b) { return sycl::isinf(bfloat162float(b)); }

bool heq(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  return __builtin_bit_cast(_iml_bfloat16_internal, b1) ==
         __builtin_bit_cast(_iml_bfloat16_internal, b2);
}

bool hequ(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b1))
    return true;
  return __builtin_bit_cast(_iml_bfloat16_internal, b1) ==
         __builtin_bit_cast(_iml_bfloat16_internal, b2);
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
  _iml_bfloat16_internal ibi = 0x7FC0;
  if (hisnan(b1) && hisnan(b2))
    return __builtin_bit_cast(sycl_bfloat16, ibi);
  if (hisnan(b1))
    return b2;
  else if (hisnan(b2))
    return b1;
  else {
    return (hgt(b1, b2) ? b1 : b2);
  }
}

sycl_bfloat16 hmax_nan(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  _iml_bfloat16_internal ibi = 0x7FC0;
  if (hisnan(b1) || hisnan(b2))
    return __builtin_bit_cast(sycl_bfloat16, ibi);
  else
    return (hgt(b1, b2) ? b1 : b2);
}

sycl_bfloat16 hmin(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  _iml_bfloat16_internal ibi = 0x7FC0;
  if (hisnan(b1) && hisnan(b2))
    return __builtin_bit_cast(sycl_bfloat16, ibi);
  if (hisnan(b1))
    return b2;
  else if (hisnan(b2))
    return b1;
  else {
    return (hlt(b1, b2) ? b1 : b2);
  }
}

sycl_bfloat16 hmin_nan(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  _iml_bfloat16_internal ibi = 0x7FC0;
  if (hisnan(b1) || hisnan(b2))
    return __builtin_bit_cast(sycl_bfloat16, ibi);
  else
    return (hlt(b1, b2) ? b1 : b2);
}

bool hne(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return false;
  return __builtin_bit_cast(_iml_bfloat16_internal, b1) !=
         __builtin_bit_cast(_iml_bfloat16_internal, b2);
}

bool hneu(sycl_bfloat16 b1, sycl_bfloat16 b2) {
  if (hisnan(b1) || hisnan(b2))
    return true;
  return __builtin_bit_cast(_iml_bfloat16_internal, b1) !=
         __builtin_bit_cast(_iml_bfloat16_internal, b2);
}
#endif
} // namespace math
} // namespace intel
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
