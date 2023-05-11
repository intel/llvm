//==--------- imf_fp_conversions.hpp - floating point conversions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// APIs for floating point conversions
//===----------------------------------------------------------------------===//

#pragma once

extern "C" {
unsigned short __imf_bfloat162ushort_rd(uint16_t);
unsigned short __imf_bfloat162ushort_rn(uint16_t);
unsigned short __imf_bfloat162ushort_ru(uint16_t);
unsigned short __imf_bfloat162ushort_rz(uint16_t);
short __imf_bfloat162short_rd(uint16_t);
short __imf_bfloat162short_rn(uint16_t);
short __imf_bfloat162short_ru(uint16_t);
short __imf_bfloat162short_rz(uint16_t);
unsigned int __imf_bfloat162uint_rd(uint16_t);
unsigned int __imf_bfloat162uint_rn(uint16_t);
unsigned int __imf_bfloat162uint_ru(uint16_t);
unsigned int __imf_bfloat162uint_rz(uint16_t);
int __imf_bfloat162int_rd(uint16_t);
int __imf_bfloat162int_rn(uint16_t);
int __imf_bfloat162int_ru(uint16_t);
int __imf_bfloat162int_rz(uint16_t);
unsigned long long __imf_bfloat162ull_rd(uint16_t);
unsigned long long __imf_bfloat162ull_rn(uint16_t);
unsigned long long __imf_bfloat162ull_ru(uint16_t);
unsigned long long __imf_bfloat162ull_rz(uint16_t);
long long __imf_bfloat162ll_rd(uint16_t);
long long __imf_bfloat162ll_rn(uint16_t);
long long __imf_bfloat162ll_ru(uint16_t);
long long __imf_bfloat162ll_rz(uint16_t);
float __imf_bfloat162float(uint16_t);
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

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>
__internal_fp_as(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat16_as_ushort(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, short>, short> __internal_fp_as(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat16_as_short(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_as(std::enable_if_t<std::is_same_v<From, short>, short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_short_as_bfloat16(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_as(
    std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ushort_as_bfloat16(x));
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat16_as_ushort(From x) {
  return __internal_fp_as<
      std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat16_as_short(From x) {
  return __internal_fp_as<
      std::enable_if_t<std::is_same_v<To, short>, short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort_as_bfloat16(From x) {
  return __internal_fp_as<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, unsigned short>, unsigned short>>(
      x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short_as_bfloat16(From x) {
  return __internal_fp_as<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, short>, short>>(x);
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ushort_rd(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ushort_rn(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ushort_ru(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ushort_rz(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, short>, short>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162short_rd(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, short>, short>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162short_rn(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, short>, short>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162short_ru(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, short>, short>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162short_rz(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned int>, unsigned int>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162uint_rd(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned int>, unsigned int>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162uint_rn(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned int>, unsigned int>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162uint_ru(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned int>, unsigned int>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162uint_rz(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, int>, int> __internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162int_rd(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, int>, int> __internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162int_rn(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, int>, int> __internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162int_ru(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, int>, int> __internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162int_rz(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned long long>,
                        unsigned long long>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ull_rd(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned long long>,
                        unsigned long long>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ull_rn(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned long long>,
                        unsigned long long>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ull_ru(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, unsigned long long>,
                        unsigned long long>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ull_rz(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, long long>, long long>
__internal_fp_convert_rd(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ll_rd(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, long long>, long long>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ll_rn(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, long long>, long long>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ll_ru(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, long long>, long long>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162ll_rz(__builtin_bit_cast(uint16_t, x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, double>, double> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_double2bfloat16(x));
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
    std::enable_if_t<std::is_same_v<From, long long>, long long> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ll2bfloat16_rd(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, long long>, long long> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ll2bfloat16_rn(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_ru(
    std::enable_if_t<std::is_same_v<From, long long>, long long> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ll2bfloat16_ru(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                        sycl::ext::oneapi::bfloat16>
__internal_fp_convert_rz(
    std::enable_if_t<std::is_same_v<From, long long>, long long> x) {
  return __builtin_bit_cast(sycl::ext::oneapi::bfloat16,
                            __imf_ll2bfloat16_rz(x));
}

template <typename To, typename From>
static std::enable_if_t<std::is_same_v<To, float>, float>
__internal_fp_convert_rn(
    std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                     sycl::ext::oneapi::bfloat16>
        x) {
  return __imf_bfloat162float(__builtin_bit_cast(uint16_t, x));
}

template <typename To = float, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162float(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, float>, float>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, float>, float>>(x);
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

template <typename To = sycl::ext::oneapi::bfloat16, typename From = long long>
To ll2bfloat16_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, long long>, long long>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = long long>
To ll2bfloat16_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, long long>, long long>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = long long>
To ll2bfloat16_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, long long>, long long>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = long long>
To ll2bfloat16_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, long long>, long long>>(x);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = double>
To double2bfloat16(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>,
      std::enable_if_t<std::is_same_v<From, double>, double>>(x);
}

template <typename To = unsigned int,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162uint_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, unsigned int>, unsigned int>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned int,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162uint_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, unsigned int>, unsigned int>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned int,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162uint_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, unsigned int>, unsigned int>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned int,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162uint_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, unsigned int>, unsigned int>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = int, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162int_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, int>, int>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = int, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162int_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, int>, int>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = int, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162int_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, int>, int>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = int, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162int_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, int>, int>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ushort_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ushort_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ushort_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ushort_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, unsigned short>, unsigned short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162short_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, short>, short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162short_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, short>, short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162short_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, short>, short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162short_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, short>, short>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = long long, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ll_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, long long>, long long>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = long long, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ll_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, long long>, long long>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = long long, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ll_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, long long>, long long>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = long long, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ll_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, long long>, long long>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned long long,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ull_rd(From x) {
  return __internal_fp_convert_rd<
      std::enable_if_t<std::is_same_v<To, unsigned long long>,
                       unsigned long long>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned long long,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ull_rn(From x) {
  return __internal_fp_convert_rn<
      std::enable_if_t<std::is_same_v<To, unsigned long long>,
                       unsigned long long>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned long long,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ull_ru(From x) {
  return __internal_fp_convert_ru<
      std::enable_if_t<std::is_same_v<To, unsigned long long>,
                       unsigned long long>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

template <typename To = unsigned long long,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ull_rz(From x) {
  return __internal_fp_convert_rz<
      std::enable_if_t<std::is_same_v<To, unsigned long long>,
                       unsigned long long>,
      std::enable_if_t<std::is_same_v<From, sycl::ext::oneapi::bfloat16>,
                       sycl::ext::oneapi::bfloat16>>(x);
}

} // namespace ext::intel::math
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
