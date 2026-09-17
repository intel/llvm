//==------ saturating_conversion.hpp - float->int8 round/saturate ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-to-nearest-even, saturating float -> 8-bit integer conversions, plus
// packed 4-wide variants that produce the byte layout consumed by the
// sycl_ext_oneapi_dot_accumulate `dot_acc` (dp4a) helpers.
//
// Semantics (identical on every backend):
//   * the float is rounded to the nearest integer using round-to-nearest-even;
//   * the result is saturated (clamped) to the destination type's range
//     ([-128, 127] for int8, [0, 255] for uint8);
//   * NaN maps to 0.
//
// This mirrors CUDA's `cvt.rni.sat.s8.f32` / `cvt.rni.sat.u8.f32`, which are
// emitted directly on NVPTX. Other targets use a portable round + clamp
// sequence (e.g. `v_rndne_f32` + `v_max_f32`/`v_min_f32` + `v_cvt_i32_f32` on
// AMDGCN), which produces bit-identical results.

#pragma once

#include <sycl/vector.hpp>

#include <cstdint>

#define SYCL_EXT_ONEAPI_SATURATING_CONVERSION 1

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace detail {

// Portable round-to-nearest-even + saturate to [lo, hi], with NaN -> 0.
// `lo`/`hi` are the inclusive bounds of the destination integer type expressed
// as exactly representable float values.
inline int32_t f_to_int_rn_sat(float x, float lo, float hi) {
  float r = __builtin_rintf(x);
  r = __builtin_fminf(__builtin_fmaxf(r, lo), hi);
  // rint(NaN) is NaN and the clamp above leaves it as `lo`, so handle NaN
  // explicitly to match the hardware `cvt.rni.sat` behavior (NaN -> 0).
  return __builtin_isnan(x) ? 0 : static_cast<int32_t>(r);
}

} // namespace detail

// float -> signed 8-bit, round-to-nearest-even, saturating, NaN -> 0.
inline int8_t float_to_int8_rn(float x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  unsigned dst;
  asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return static_cast<int8_t>(dst);
#else
  return static_cast<int8_t>(detail::f_to_int_rn_sat(x, -128.0f, 127.0f));
#endif
}

// float -> unsigned 8-bit, round-to-nearest-even, saturating, NaN -> 0.
inline uint8_t float_to_uint8_rn(float x) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  unsigned dst;
  asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return static_cast<uint8_t>(dst);
#else
  return static_cast<uint8_t>(detail::f_to_int_rn_sat(x, 0.0f, 255.0f));
#endif
}

// Convert four floats to signed int8 and pack them into a 32-bit word, with
// lane i occupying byte i. The layout matches `dot_acc(vec<int8_t, 4>, ...)`,
// so the result can be reinterpreted and fed directly to dp4a.
inline int32_t float4_to_int8x4_rn(vec<float, 4> v) {
  uint32_t r = static_cast<uint8_t>(float_to_int8_rn(v.s0()));
  r |= static_cast<uint32_t>(static_cast<uint8_t>(float_to_int8_rn(v.s1()))) << 8;
  r |= static_cast<uint32_t>(static_cast<uint8_t>(float_to_int8_rn(v.s2())))
       << 16;
  r |= static_cast<uint32_t>(static_cast<uint8_t>(float_to_int8_rn(v.s3())))
       << 24;
  return static_cast<int32_t>(r);
}

// Convert four floats to unsigned uint8 and pack them into a 32-bit word, with
// lane i occupying byte i.
inline uint32_t float4_to_uint8x4_rn(vec<float, 4> v) {
  uint32_t r = float_to_uint8_rn(v.s0());
  r |= static_cast<uint32_t>(float_to_uint8_rn(v.s1())) << 8;
  r |= static_cast<uint32_t>(float_to_uint8_rn(v.s2())) << 16;
  r |= static_cast<uint32_t>(float_to_uint8_rn(v.s3())) << 24;
  return r;
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
