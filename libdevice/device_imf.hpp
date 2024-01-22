//==------- device_imf.h - intel math devicelib functions declarations------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __LIBDEVICE_DEVICE_IMF_H__
#define __LIBDEVICE_DEVICE_IMF_H__

#include "device.h"
#include "imf_bf16.hpp"
#include "imf_half.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#ifdef __LIBDEVICE_IMF_ENABLED__

#if !defined(__SPIR__) && !defined(__LIBDEVICE_HOST_IMPL__)
#error                                                                         \
    "__SPIR__ or __LIBDEVICE_HOST_IMPL__ must be defined to enable device imf functions!"
#endif

// TODO: Bitcast is valid to trivially copyable object only but using
// is_trivially_copyable check will lead to compiling error in some
// pre-ci tests, the pre-ci environment used some legacy c++ std library
// which doesn't include this function. Need to report to pre-ci owners.
template <typename To, typename From>
static inline constexpr To __bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From),
                "Can't do bit cast between 2 types with different sizes!");
  /*static_assert(std::is_trivially_copyable<From>::value &&
                    std::is_trivially_copyable<To>::value,
                "Can't do bit cast for type which is not trivially
     copyable!");*/
  return __builtin_bit_cast(To, from);
}

#if defined(__LIBDEVICE_HOST_IMPL__)
#include <cfenv>
#pragma STDC FENV_ACCESS ON

template <typename Tp> static inline Tp __double2Tp_host(double x, int rdMode) {
  static_assert(std::is_same<Tp, float>::value ||
                    std::is_same<Tp, int>::value ||
                    std::is_same<Tp, unsigned int>::value ||
                    std::is_same<Tp, long long int>::value ||
                    std::is_same<Tp, unsigned long long int>::value,
                "Invalid type for double conversion!");

  const int roundingOriginal = fegetround();
  fesetround(rdMode);
  Tp res;
  if (std::is_same<Tp, float>::value)
    res = static_cast<float>(x);
  else
    res = static_cast<Tp>(__builtin_nearbyint(x));
  fesetround(roundingOriginal);
  return res;
}

template <typename Tp> static inline Tp __float2Tp_host(float x, int rdMode) {
  static_assert(std::is_same<Tp, int>::value ||
                    std::is_same<Tp, unsigned int>::value ||
                    std::is_same<Tp, long long int>::value ||
                    std::is_same<Tp, unsigned long long int>::value,
                "Invalid type for float conversion!");

  const int roundingOriginal = fegetround();
  fesetround(rdMode);
  Tp res = static_cast<Tp>(__builtin_nearbyintf(x));
  fesetround(roundingOriginal);
  return res;
}

template <typename TyINT, typename TyFP>
static inline TyFP __integral2FP_host(TyINT x, int rdMode) {
  static_assert((std::is_same<TyINT, int>::value ||
                 std::is_same<TyINT, unsigned int>::value ||
                 std::is_same<TyINT, long>::value ||
                 std::is_same<TyINT, unsigned long>::value ||
                 std::is_same<TyINT, long long int>::value ||
                 std::is_same<TyINT, unsigned long long int>::value) &&
                    (std::is_same<TyFP, float>::value ||
                     std::is_same<TyFP, double>::value),
                "Invalid integral to FP conversion!");
  const int roundingOriginal = fegetround();
  fesetround(rdMode);
  TyFP res = static_cast<TyFP>(x);
  fesetround(roundingOriginal);
  return res;
}
#pragma STDC FENV_ACCESS OFF
#endif // __LIBDEVICE_HOST_IMPL__

template <typename Ty> static inline Ty __imax(Ty x, Ty y) {
  static_assert(std::is_integral<Ty>::value,
                "__imax only accepts integral type.");
  return (x > y) ? x : y;
}

template <typename Ty> static inline Ty __imin(Ty x, Ty y) {
  static_assert(std::is_integral<Ty>::value,
                "__imin only accepts integral type.");
  return (x < y) ? x : y;
}

static inline float __fclamp(float x, float y, float z) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fmin(__builtin_fmax(x, y), z);
#elif defined(__SPIR__)
  return __spirv_ocl_fclamp(x, y, z);
#endif
}

// fma for float, double, half, bf16 math, covers both device and host.
static inline float __fma(float x, float y, float z) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fmaf(x, y, z);
#elif defined(__SPIR__)
  return __spirv_ocl_fma(x, y, z);
#endif
}

static inline double __fma(double x, double y, double z) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fma(x, y, z);
#elif defined(__SPIR__)
  return __spirv_ocl_fma(x, y, z);
#endif
}

static inline _iml_half __fma(_iml_half x, _iml_half y, _iml_half z) {
  _iml_half_internal x_i = x.get_internal();
  _iml_half_internal y_i = y.get_internal();
  _iml_half_internal z_i = z.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float tmp_y = __half2float(y_i);
  float tmp_z = __half2float(z_i);
  float res = __builtin_fmaf(tmp_x, tmp_y, tmp_z);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_fma(x_i, y_i, z_i));
#endif
}

// Currently, we used fp32 to emulate all bf16 arithmetic
static inline _iml_bf16 __fma(_iml_bf16 x, _iml_bf16 y, _iml_bf16 z) {
  float tmp_x = __bfloat162float(x.get_internal());
  float tmp_y = __bfloat162float(y.get_internal());
  float tmp_z = __bfloat162float(z.get_internal());
  float res = __fma(tmp_x, tmp_y, tmp_z);
  return _iml_bf16(res);
}

// sqrt for float, double, half, bf16 math, covers both device and host.
static inline float __sqrt(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_sqrtf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_sqrt(x);
#endif
}

static inline double __sqrt(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_sqrt(x);
#elif defined(__SPIR__)
  return __spirv_ocl_sqrt(x);
#endif
}

static inline _iml_half __sqrt(_iml_half x) {
  _iml_half_internal x_i = x.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float res = __builtin_sqrtf(tmp_x);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_sqrt(x_i));
#endif
}

static inline _iml_bf16 __sqrt(_iml_bf16 x) {
  float tmp_x = __bfloat162float(x.get_internal());
  float res = __sqrt(tmp_x);
  return _iml_bf16(res);
}

// rsqrt for float, double, half, bf16 math, covers both device and host.
static inline float __rsqrt(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return 1.f / __builtin_sqrtf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_rsqrt(x);
#endif
}

static inline double __rsqrt(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return 1.0 / __builtin_sqrt(x);
#elif defined(__SPIR__)
  return __spirv_ocl_rsqrt(x);
#endif
}

static inline _iml_half __rsqrt(_iml_half x) {
  _iml_half_internal x_i = x.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float res = 1.f / __builtin_sqrtf(tmp_x);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_rsqrt(x_i));
#endif
}

static inline _iml_bf16 __rsqrt(_iml_bf16 x) {
  float tmp_x = __bfloat162float(x.get_internal());
  float res = __rsqrt(tmp_x);
  return _iml_bf16(res);
}

// fmin for float, double, half, bf16 math, covers both device and host.
static inline float __fmin(float x, float y) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fminf(x, y);
#elif defined(__SPIR__)
  return __spirv_ocl_fmin(x, y);
#endif
}

static inline double __fmin(double x, double y) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fmin(x, y);
#elif defined(__SPIR__)
  return __spirv_ocl_fmin(x, y);
#endif
}

static inline _iml_half __fmin(_iml_half x, _iml_half y) {
  _iml_half_internal x_i = x.get_internal();
  _iml_half_internal y_i = y.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float tmp_y = __half2float(y_i);
  float res = __builtin_fminf(tmp_x, tmp_y);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_fmin(x_i, y_i));
#endif
}

static inline _iml_bf16 __fmin(_iml_bf16 x, _iml_bf16 y) {
  float tmp_x = __bfloat162float(x.get_internal());
  float tmp_y = __bfloat162float(y.get_internal());
  float res = __fmin(tmp_x, tmp_y);
  return _iml_bf16(res);
}

// fmax for float, double, half, bf16 math, covers both device and host.
static inline float __fmax(float x, float y) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fmaxf(x, y);
#elif defined(__SPIR__)
  return __spirv_ocl_fmax(x, y);
#endif
}

static inline double __fmax(double x, double y) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fmax(x, y);
#elif defined(__SPIR__)
  return __spirv_ocl_fmax(x, y);
#endif
}

static inline _iml_half __fmax(_iml_half x, _iml_half y) {
  _iml_half_internal x_i = x.get_internal();
  _iml_half_internal y_i = y.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float tmp_y = __half2float(y_i);
  float res = __builtin_fmaxf(tmp_x, tmp_y);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_fmax(x_i, y_i));
#endif
}

static inline _iml_bf16 __fmax(_iml_bf16 x, _iml_bf16 y) {
  float tmp_x = __bfloat162float(x.get_internal());
  float tmp_y = __bfloat162float(y.get_internal());
  float res = __fmax(tmp_x, tmp_y);
  return _iml_bf16(res);
}

// copysign for float, double, half, bf16 math, covers both device and host.
static inline float __copysign(float x, float y) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_copysignf(x, y);
#elif defined(__SPIR__)
  return __spirv_ocl_copysign(x, y);
#endif
}

static inline double __copysign(double x, double y) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_copysign(x, y);
#elif defined(__SPIR__)
  return __spirv_ocl_copysign(x, y);
#endif
}

static inline _iml_half __copysign(_iml_half x, _iml_half y) {
  _iml_half_internal x_i = x.get_internal();
  _iml_half_internal y_i = y.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float tmp_y = __half2float(y_i);
  float res = __builtin_copysignf(tmp_x, tmp_y);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_copysign(x_i, y_i));
#endif
}

static inline _iml_bf16 __copysign(_iml_bf16 x, _iml_bf16 y) {
  float tmp_x = __bfloat162float(x.get_internal());
  float tmp_y = __bfloat162float(y.get_internal());
  float res = __copysign(tmp_x, tmp_y);
  return _iml_bf16(res);
}

// fabs for float, double, half, bf16 math, covers both device and host.
static inline float __fabs(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fabsf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_fabs(x);
#endif
}

static inline double __fabs(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_fabs(x);
#elif defined(__SPIR__)
  return __spirv_ocl_fabs(x);
#endif
}

static inline _iml_half __fabs(_iml_half x) {
  _iml_half_internal x_i = x.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float res = __builtin_fabsf(tmp_x);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_fabs(x_i));
#endif
}

static inline _iml_bf16 __fabs(_iml_bf16 x) {
  float tmp_x = __bfloat162float(x.get_internal());
  float res = __fabs(tmp_x);
  return _iml_bf16(res);
}

// rint for float, double, half, bf16 math, covers both device and host.
static inline float __rint(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_rintf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_rint(x);
#endif
}

static inline double __rint(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_rint(x);
#elif defined(__SPIR__)
  return __spirv_ocl_rint(x);
#endif
}

static inline _iml_half __rint(_iml_half x) {
  _iml_half_internal x_i = x.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float res = __builtin_rintf(tmp_x);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_rint(x_i));
#endif
}

static inline _iml_bf16 __rint(_iml_bf16 x) {
  float tmp_x = __bfloat162float(x.get_internal());
  float res = __rint(tmp_x);
  return _iml_bf16(res);
}

// floor for float, double, half, bf16 math, covers both device and host.
static inline float __floor(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_floorf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_floor(x);
#endif
}

static inline double __floor(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_floor(x);
#elif defined(__SPIR__)
  return __spirv_ocl_floor(x);
#endif
}

static inline _iml_half __floor(_iml_half x) {
  _iml_half_internal x_i = x.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float res = __builtin_floorf(tmp_x);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_floor(x_i));
#endif
}

static inline _iml_bf16 __floor(_iml_bf16 x) {
  float tmp_x = __bfloat162float(x.get_internal());
  float res = __floor(tmp_x);
  return _iml_bf16(res);
}

// ceil for float, double, half, bf16 math, covers both device and host.
static inline float __ceil(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_ceilf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_ceil(x);
#endif
}

static inline double __ceil(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_ceil(x);
#elif defined(__SPIR__)
  return __spirv_ocl_ceil(x);
#endif
}

static inline _iml_half __ceil(_iml_half x) {
  _iml_half_internal x_i = x.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float res = __builtin_ceilf(tmp_x);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_ceil(x_i));
#endif
}

static inline _iml_bf16 __ceil(_iml_bf16 x) {
  float tmp_x = __bfloat162float(x.get_internal());
  float res = __ceil(tmp_x);
  return _iml_bf16(res);
}

// trunc for float, double, half, bf16 math, covers both device and host.
static inline float __trunc(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_truncf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_trunc(x);
#endif
}

static inline double __trunc(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_trunc(x);
#elif defined(__SPIR__)
  return __spirv_ocl_trunc(x);
#endif
}

static inline float __fast_exp10f(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_expf(0x1.26bb1cp1f * x);
#elif defined(__SPIR__)
  return __spirv_ocl_native_exp(0x1.26bb1cp1f * x);
#endif
}

static inline float __fast_expf(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_expf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_native_exp(x);
#endif
}

static inline float __fast_logf(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_logf(x);
#elif defined(__SPIR__)
  return __spirv_ocl_native_log(x);
#endif
}

static inline float __fast_log2f(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_log2f(x);
#elif defined(__SPIR__)
  return __spirv_ocl_native_log(x) / 0x1.62e43p-1f;
#endif
}

static inline float __fast_log10f(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_log10f(x);
#elif defined(__SPIR__)
  return __spirv_ocl_native_log(x) / 0x1.26bb1cp1f;
#endif
}

static inline float __fast_powf(float x, float y) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_powf(x, y);
#elif defined(__SPIR__)
  return __spirv_ocl_native_powr(x, y);
#endif
}

static inline float __fast_fdividef(float x, float y) {
  unsigned ybits = __builtin_bit_cast(unsigned, y);
  unsigned xbits = __builtin_bit_cast(unsigned, x);
  ybits &= 0x7FFF'FFFF;
  xbits &= 0x7FFF'FFFF;
  unsigned yexp_bits = (ybits >> 23) & 0xFF;
  unsigned xexp_bits = (xbits >> 23) & 0xFF;
  unsigned yman_bits = ybits & 0x7F'FFFF;
  unsigned xman_bits = xbits & 0x7F'FFFF;
  if (ybits > 0x7E80'0000) {
    if ((xexp_bits = 0xFF) && (xman_bits == 0))
      return __builtin_bit_cast(float, 0x7FC00000);
    else
      return 0;
  }

#if defined(__LIBDEVICE_HOST_IMPL__)
  return x / y;
#elif defined(__SPIR__)
  return __spirv_ocl_native_divide(x, y);
#endif
}

static inline _iml_half __trunc(_iml_half x) {
  _iml_half_internal x_i = x.get_internal();
#if defined(__LIBDEVICE_HOST_IMPL__)
  float tmp_x = __half2float(x_i);
  float res = __builtin_truncf(tmp_x);
  return _iml_half(__float2half(res));
#elif defined(__SPIR__)
  return _iml_half(__spirv_ocl_trunc(x_i));
#endif
}

static inline _iml_bf16 __trunc(_iml_bf16 x) {
  float tmp_x = __bfloat162float(x.get_internal());
  float res = __trunc(tmp_x);
  return _iml_bf16(res);
}

static inline int __clz(int x) {
  if (x == 0)
    return 32;
  uint32_t xi32 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_clz(xi32);
#elif defined(__SPIR__)
  return __spirv_ocl_clz(xi32);
#endif
}

static inline int __clzll(long long int x) {
  if (x == 0)
    return 64;
  uint64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_clzll(xi64);
#elif defined(__SPIR__)
  return __spirv_ocl_clz(xi64);
#endif
}

static inline int __popc(unsigned int x) {
  uint32_t xui32 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_popcount(xui32);
#elif defined(__SPIR__)
  return __spirv_ocl_popcount(xui32);
#endif
}

static inline int __popcll(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __builtin_popcountll(xui64);
#elif defined(__SPIR__)
  return __spirv_ocl_popcount(xui64);
#endif
}

template <typename T>
static inline typename std::make_unsigned<T>::type __abs(T x) {
  static_assert((std::is_signed<T>::value && std::is_integral<T>::value),
                "__abs can only accept signed integral type.");
  return x < 0 ? -x : x;
}

template <typename T> static inline void __swap(T &x, T &y) {
  static_assert(std::is_integral<T>::value,
                "__swap can only accept integral type.");
  T tmp = x;
  x = y;
  y = tmp;
}

template <typename Ty1, typename Ty2>
static inline Ty2 __get_bytes_by_index(Ty1 x, size_t idx) {
  static_assert(!std::is_signed<Ty1>::value && !std::is_signed<Ty2>::value,
                "__get_bytes_by_index can only accept unsigned value.");
  static_assert(std::is_integral<Ty1>::value && std::is_integral<Ty2>::value,
                "__get_bytes_by_index can only accept integral type.");
  size_t bits_shift = idx * sizeof(Ty2) * 8;
  Ty1 mask1 = static_cast<Ty1>(-1);
  x >>= bits_shift;
  x = x & mask1;
  return static_cast<Ty2>(x);
}

template <typename Ty1, typename Ty2, size_t N>
Ty1 __assemble_integral_value(Ty2 *x) {
  static_assert(!std::is_signed<Ty1>::value && !std::is_signed<Ty2>::value,
                "__assemble_integeral_value can only accept unsigned value.");
  static_assert(std::is_integral<Ty1>::value && std::is_integral<Ty2>::value,
                "__assemble_integeral_value can only accept integral value.");
  static_assert(sizeof(Ty1) == N * sizeof(Ty2),
                "size mismatch for __assemble_integeral_value");
  Ty1 res = 0;
  for (size_t idx = 0; idx < N; ++idx) {
    res <<= sizeof(Ty2) * 8;
    res |= static_cast<Ty1>(x[N - 1 - idx]);
  }
  return res;
}

template <typename Ty> static inline Ty __uhadd(Ty x, Ty y) {
  static_assert(std::is_integral<Ty>::value && !std::is_signed<Ty>::value,
                "__uhadd can only accept unsigned integral type.");
#if defined(__LIBDEVICE_HOST_IMPL__)
  return (x >> 1) + (y >> 1) + ((x & y) & 0x1);
#elif defined(__SPIR__)
  return __spirv_ocl_u_hadd(x, y);
#endif
}

template <typename Ty> static inline Ty __shadd(Ty x, Ty y) {
  static_assert(std::is_integral<Ty>::value && std::is_signed<Ty>::value,
                "__shadd can only accept signed integral type.");
#if defined(__LIBDEVICE_HOST_IMPL__)
  return (x >> 1) + (y >> 1) + ((x & y) & 0x1);
#elif defined(__SPIR__)
  return __spirv_ocl_s_hadd(x, y);
#endif
}

template <typename Ty> static inline Ty __urhadd(Ty x, Ty y) {
  static_assert(std::is_integral<Ty>::value && !std::is_signed<Ty>::value,
                "__urhadd can only accept unsigned integral type.");
#if defined(__LIBDEVICE_HOST_IMPL__)
  return (x >> 1) + (y >> 1) + ((x | y) & 0x1);
#elif defined(__SPIR__)
  return __spirv_ocl_u_rhadd(x, y);
#endif
}

template <typename Ty> static inline Ty __srhadd(Ty x, Ty y) {
  static_assert(std::is_integral<Ty>::value && std::is_signed<Ty>::value,
                "__srhadd can only accept signed integral type.");
#if defined(__LIBDEVICE_HOST_IMPL__)
  return (x >> 1) + (y >> 1) + ((x | y) & 0x1);
#elif defined(__SPIR__)
  return __spirv_ocl_s_rhadd(x, y);
#endif
}
#endif // __LIBDEVICE_IMF_ENABLED__
#endif // __LIBDEVICE_DEVICE_IMF_H__
