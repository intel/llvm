//==-------------- math.hpp - Intel specific math API ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The main header of Intel specific math API
//===----------------------------------------------------------------------===//

#pragma once

// _iml_half_internal is internal representation for fp16 type used in intel
// math device library. The definition here should align with definition in
// https://github.com/intel/llvm/blob/sycl/libdevice/imf_half.hpp
#if defined(__SPIR__)
using _iml_half_internal = _Float16;
#else
using _iml_half_internal = uint16_t;
#endif

#include <sycl/builtins.hpp>
#include <sycl/ext/intel/math/imf_fp_conversions.hpp>
#include <sycl/ext/intel/math/imf_half_trivial.hpp>
#include <sycl/ext/intel/math/imf_rounding_math.hpp>
#include <sycl/ext/intel/math/imf_simd.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>
#include <type_traits>

extern "C" {
float __imf_saturatef(float);
float __imf_copysignf(float, float);
double __imf_copysign(double, double);
_iml_half_internal __imf_copysignf16(_iml_half_internal, _iml_half_internal);
float __imf_ceilf(float);
double __imf_ceil(double);
_iml_half_internal __imf_ceilf16(_iml_half_internal);
float __imf_floorf(float);
double __imf_floor(double);
_iml_half_internal __imf_floorf16(_iml_half_internal);
float __imf_rintf(float);
double __imf_rint(double);
_iml_half_internal __imf_invf16(_iml_half_internal);
float __imf_invf(float);
double __imf_inv(double);
_iml_half_internal __imf_rintf16(_iml_half_internal);
float __imf_sqrtf(float);
double __imf_sqrt(double);
_iml_half_internal __imf_sqrtf16(_iml_half_internal);
float __imf_rsqrtf(float);
double __imf_rsqrt(double);
_iml_half_internal __imf_rsqrtf16(_iml_half_internal);
float __imf_truncf(float);
double __imf_trunc(double);
_iml_half_internal __imf_truncf16(_iml_half_internal);
double __imf_rcp64h(double);
};

namespace sycl {
inline namespace _V1 {
namespace ext::intel::math {

static_assert(sizeof(sycl::half) == sizeof(_iml_half_internal),
              "sycl::half is not compatible with _iml_half_internal.");

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> saturate(Tp x) {
  return __imf_saturatef(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> copysign(Tp x, Tp y) {
  return __imf_copysignf(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> copysign(Tp x, Tp y) {
  return __imf_copysign(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> copysign(Tp x,
                                                                      Tp y) {
  _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
  _iml_half_internal yi = __builtin_bit_cast(_iml_half_internal, y);
  return __builtin_bit_cast(sycl::half, __imf_copysignf16(xi, yi));
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> ceil(Tp x) {
  return __imf_ceilf(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> ceil(Tp x) {
  return __imf_ceil(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> ceil(Tp x) {
  _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
  return __builtin_bit_cast(sycl::half, __imf_ceilf16(xi));
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> ceil(Tp x) {
  return sycl::half2{ceil(x.s0()), ceil(x.s1())};
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> floor(Tp x) {
  return __imf_floorf(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> floor(Tp x) {
  return __imf_floor(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> floor(Tp x) {
  _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
  return __builtin_bit_cast(sycl::half, __imf_floorf16(xi));
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> floor(Tp x) {
  return sycl::half2{floor(x.s0()), floor(x.s1())};
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> inv(Tp x) {
  return __imf_invf(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> inv(Tp x) {
  return __imf_inv(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> inv(Tp x) {
  _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
  return __builtin_bit_cast(sycl::half, __imf_invf16(xi));
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> inv(Tp x) {
  return sycl::half2{inv(x.s0()), inv(x.s1())};
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> rint(Tp x) {
  return __imf_rintf(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> rint(Tp x) {
  return __imf_rint(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> rint(Tp x) {
  _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
  return __builtin_bit_cast(sycl::half, __imf_rintf16(xi));
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> rint(Tp x) {
  return sycl::half2{rint(x.s0()), rint(x.s1())};
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> sqrt(Tp x) {
  return __imf_sqrtf(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> sqrt(Tp x) {
  return __imf_sqrt(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> sqrt(Tp x) {
  _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
  return __builtin_bit_cast(sycl::half, __imf_sqrtf16(xi));
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> sqrt(Tp x) {
  return sycl::half2{sqrt(x.s0()), sqrt(x.s1())};
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> rsqrt(Tp x) {
  return __imf_rsqrtf(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> rsqrt(Tp x) {
  return __imf_rsqrt(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> rsqrt(Tp x) {
  _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
  return __builtin_bit_cast(sycl::half, __imf_rsqrtf16(xi));
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> rsqrt(Tp x) {
  return sycl::half2{rsqrt(x.s0()), rsqrt(x.s1())};
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, float>, float> trunc(Tp x) {
  return __imf_truncf(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> trunc(Tp x) {
  return __imf_trunc(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> trunc(Tp x) {
  _iml_half_internal xi = __builtin_bit_cast(_iml_half_internal, x);
  return __builtin_bit_cast(sycl::half, __imf_truncf16(xi));
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> trunc(Tp x) {
  return sycl::half2{trunc(x.s0()), trunc(x.s1())};
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> rcp64h(Tp x) {
  return __imf_rcp64h(x);
}

} // namespace ext::intel::math
} // namespace _V1
} // namespace sycl
