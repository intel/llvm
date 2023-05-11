//==------------- math.hpp - Intel specific math API-----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The main header of Intel specific math API
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/builtins.hpp>
#include <sycl/ext/intel/math/imf_half_trivial.hpp>
#include <sycl/ext/intel/math/imf_simd.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>
#include <type_traits>

// _iml_half_internal is internal representation for fp16 type used in intel
// math device library. The definition here should align with definition in
// https://github.com/intel/llvm/blob/sycl/libdevice/imf_half.hpp
#if defined(__SPIR__)
using _iml_half_internal = _Float16;
#else
using _iml_half_internal = uint16_t;
#endif

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
// bfloat16 conversions using software emulation
float __imf_bfloat162float(uint16_t x);
unsigned int __imf_bfloat162uint_rd(uint16_t);
unsigned int __imf_bfloat162uint_rn(uint16_t);
unsigned int __imf_bfloat162uint_ru(uint16_t);
unsigned int __imf_bfloat162uint_rz(uint16_t);
unsigned short __imf_bfloat162ushort_rd(uint16_t);
unsigned short __imf_bfloat162ushort_rn(uint16_t);
unsigned short __imf_bfloat162ushort_ru(uint16_t);
unsigned short __imf_bfloat162ushort_rz(uint16_t);
unsigned long long __imf_bfloat162ull_rd(uint16_t);
unsigned long long __imf_bfloat162ull_rn(uint16_t);
unsigned long long __imf_bfloat162ull_ru(uint16_t);
unsigned long long __imf_bfloat162ull_rz(uint16_t);
int __imf_bfloat162int_rd(uint16_t);
int __imf_bfloat162int_rn(uint16_t);
int __imf_bfloat162int_ru(uint16_t);
int __imf_bfloat162int_rz(uint16_t);
short __imf_bfloat162short_rd(uint16_t);
short __imf_bfloat162short_rn(uint16_t);
short __imf_bfloat162short_ru(uint16_t);
short __imf_bfloat162short_rz(uint16_t);
long long __imf_bfloat162ll_rd(uint16_t);
long long __imf_bfloat162ll_rn(uint16_t);
long long __imf_bfloat162ll_ru(uint16_t);
long long __imf_bfloat162ll_rz(uint16_t);
uint16_t __imf_float2bfloat16(float);
uint16_t __imf_float2bfloat16_rd(float);
uint16_t __imf_float2bfloat16_rn(float);
uint16_t __imf_float2bfloat16_ru(float);
uint16_t __imf_float2bfloat16_rz(float);
uint16_t __imf_ushort2bfloat16_rd(unsigned short);
uint16_t __imf_ushort2bfloat16_rn(unsigned short);
uint16_t __imf_ushort2bfloat16_ru(unsigned short);
uint16_t __imf_ushort2bfloat16_rz(unsigned short);
uint16_t __imf_uint2bfloat16_rd(unsigned int);
uint16_t __imf_uint2bfloat16_rn(unsigned int);
uint16_t __imf_uint2bfloat16_ru(unsigned int);
uint16_t __imf_uint2bfloat16_rz(unsigned int);
uint16_t __imf_ull2bfloat16_rd(unsigned long long);
uint16_t __imf_ull2bfloat16_rn(unsigned long long);
uint16_t __imf_ull2bfloat16_ru(unsigned long long);
uint16_t __imf_ull2bfloat16_rz(unsigned long long);
uint16_t __imf_short2bfloat16_rd(short);
uint16_t __imf_short2bfloat16_rn(short);
uint16_t __imf_short2bfloat16_ru(short);
uint16_t __imf_short2bfloat16_rz(short);
uint16_t __imf_int2bfloat16_rd(int);
uint16_t __imf_int2bfloat16_rn(int);
uint16_t __imf_int2bfloat16_ru(int);
uint16_t __imf_int2bfloat16_rz(int);
uint16_t __imf_ll2bfloat16_rd(long long);
uint16_t __imf_ll2bfloat16_rn(long long);
uint16_t __imf_ll2bfloat16_ru(long long);
uint16_t __imf_ll2bfloat16_rz(long long);
uint16_t __imf_double2bfloat16(double);
short __imf_bfloat16_as_short(uint16_t);
unsigned short __imf_bfloat16_as_ushort(uint16_t);
uint16_t __imf_short_as_bfloat16(short);
uint16_t __imf_ushort_as_bfloat16(unsigned short);
};

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, float>, float> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16_rd(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, float>, float> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16_rn(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, float>, float> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16_ru(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, float>, float> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_float2bfloat16_rz(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ushort2bfloat16_rd(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ushort2bfloat16_rn(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ushort2bfloat16_ru(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ushort2bfloat16_rz(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, unsigned int>, unsigned int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_uint2bfloat16_rd(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, unsigned int>, unsigned int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_uint2bfloat16_rn(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, unsigned int>, unsigned int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_uint2bfloat16_ru(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, unsigned int>, unsigned int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_uint2bfloat16_rz(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, unsigned long long>,
                     unsigned long long>
        x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ull2bfloat16_rd(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, unsigned long long>,
                     unsigned long long>
        x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ull2bfloat16_rn(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, unsigned long long>,
                     unsigned long long>
        x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ull2bfloat16_ru(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, unsigned long long>,
                     unsigned long long>
        x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ull2bfloat16_rz(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, short>, short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_short2bfloat16_rd(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, short>, short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_short2bfloat16_rn(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, short>, short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_short2bfloat16_ru(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, short>, short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_short2bfloat16_rz(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rd(std::enable_if_t<std::is_same_v<From, int>, int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_int2bfloat16_rd(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(std::enable_if_t<std::is_same_v<From, int>, int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_int2bfloat16_rn(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_ru(std::enable_if_t<std::is_same_v<From, int>, int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_int2bfloat16_ru(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rz(std::enable_if_t<std::is_same_v<From, int>, int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_int2bfloat16_rz(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, long long int>, long long int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ll2bfloat16_rd(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, long long int>, long long int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ll2bfloat16_rn(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, long long int>, long long int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ll2bfloat16_ru(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, long long int>, long long int> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ll2bfloat16_rz(x));
}

float bfloat162float(sycl::ext::oneapi::bfloat16 x) {
  return __imf_bfloat162float(__builtin_bit_cast(uint16_t, x));
}

template <typename Tp = sycl::ext::oneapi::bfloat16>
Tp float2bfloat16(float x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<Tp, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      float>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, float>, float>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, float>, float>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, float>, float>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, float>, float>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort2bfloat16_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short>>(
      x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort2bfloat16_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short>>(
      x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort2bfloat16_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short>>(
      x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort2bfloat16_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short>>(
      x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned int>
To uint2bfloat16_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned int>, unsigned int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned int>
To uint2bfloat16_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned int>, unsigned int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned int>
To uint2bfloat16_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned int>, unsigned int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned int>
To uint2bfloat16_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned int>, unsigned int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned long long>
To ull2bfloat16_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned long long>,
                       unsigned long long>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned long long>
To ull2bfloat16_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned long long>,
                       unsigned long long>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned long long>
To ull2bfloat16_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned long long>,
                       unsigned long long>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned long long>
To ull2bfloat16_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned long long>,
                       unsigned long long>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short2bfloat16_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, short>, short>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short2bfloat16_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, short>, short>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short2bfloat16_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, short>, short>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short2bfloat16_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, short>, short>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = int>
To int2bfloat16_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, int>, int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = int>
To int2bfloat16_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, int>, int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = int>
To int2bfloat16_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, int>, int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = int>
To int2bfloat16_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, int>, int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = long long int>
To ll2bfloat16_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, long long int>, long long int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = long long int>
To ll2bfloat16_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, long long int>, long long int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = long long int>
To ll2bfloat16_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, long long int>, long long int>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = long long int>
To ll2bfloat16_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, long long int>, long long int>>(x);
}
} // namespace ext::intel::math
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
