//==---------- builtins_integer.cpp - SYCL built-in integer functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines the host versions of functions defined
// in SYCL SPEC section - 4.13.4 Integer functions.

#include "builtins_helper.hpp"

#include <algorithm>
#include <type_traits>

namespace s = cl::sycl;
namespace d = s::detail;

namespace cl {
namespace __host_std {
namespace {

template <typename T> inline T __abs_diff(T x, T y) {
  static_assert(std::is_integral<T>::value,
                "Only integral types are supported");
  return (x > y) ? (x - y) : (y - x);
}

template <typename T> inline T __u_add_sat(T x, T y) {
  return (x < (d::max_v<T>() - y) ? x + y : d::max_v<T>());
}

template <typename T> inline T __s_add_sat(T x, T y) {
  if (x > 0 && y > 0)
    return (x < (d::max_v<T>() - y) ? (x + y) : d::max_v<T>());
  if (x < 0 && y < 0)
    return (x > (d::min_v<T>() - y) ? (x + y) : d::min_v<T>());
  return x + y;
}

template <typename T> inline T __hadd(T x, T y) {
  const T one = 1;
  return (x >> one) + (y >> one) + ((y & x) & one);
}

template <typename T> inline T __rhadd(T x, T y) {
  const T one = 1;
  return (x >> one) + (y >> one) + ((y | x) & one);
}

template <typename T> inline T __clamp(T x, T minval, T maxval) {
  return std::min(std::max(x, minval), maxval);
}

template <typename T> inline constexpr T __clz_impl(T x, T m, T n = 0) {
  return (x & m) ? n : __clz_impl(x, T(m >> 1), ++n);
}

template <typename T> inline constexpr T __clz(T x) {
  using UT = typename std::make_unsigned<T>::type;
  return (x == T(0)) ? sizeof(T) * 8 : __clz_impl<UT>(x, d::msbMask<UT>(x));
}

template <typename T> T __mul_hi(T a, T b) {
  using UPT = typename d::make_upper<T>::type;
  UPT a_s = a;
  UPT b_s = b;
  UPT mul = a_s * b_s;
  return (mul >> (sizeof(T) * 8));
}

// T is minimum of 64 bits- long or longlong
template <typename T> inline T __long_mul_hi(T a, T b) {
  int halfsize = (sizeof(T) * 8) / 2;
  T a1 = a >> halfsize;
  T a0 = (a << halfsize) >> halfsize;
  T b1 = b >> halfsize;
  T b0 = (b << halfsize) >> halfsize;

  // a1b1 - for bits - [64-128)
  // a1b0 a0b1 for bits - [32-96)
  // a0b0 for bits - [0-64)
  T a1b1 = a1 * b1;
  T a0b1 = a0 * b1;
  T a1b0 = a1 * b0;
  T a0b0 = a0 * b0;

  // To get the upper 64 bits:
  // 64 bits from a1b1, upper 32 bits from [a1b0 + (a0b1 + a0b0>>32 (carry bit
  // in 33rd bit))] with carry bit on 64th bit - use of hadd. Add the a1b1 to
  // the above 32 bit result.
  T result =
      a1b1 + (__hadd(a1b0, (a0b1 + (a0b0 >> halfsize))) >> (halfsize - 1));
  return result;
}

template <typename T> inline T __mad_hi(T a, T b, T c) {
  return __mul_hi(a, b) + c;
}

template <typename T> inline T __long_mad_hi(T a, T b, T c) {
  return __long_mul_hi(a, b) + c;
}

template <typename T> inline T __s_mad_sat(T a, T b, T c) {
  using UPT = typename d::make_upper<T>::type;
  UPT mul = UPT(a) * UPT(b);
  const UPT max = d::max_v<T>();
  const UPT min = d::min_v<T>();
  mul = std::min(std::max(mul, min), max);
  return __s_add_sat(T(mul), c);
}

template <typename T> inline T __s_long_mad_sat(T a, T b, T c) {
  bool neg_prod = (a < 0) ^ (b < 0);
  T mulhi = __long_mul_hi(a, b);

  // check mul_hi. If it is any value != 0.
  // if prod is +ve, any value in mulhi means we need to saturate.
  // if prod is -ve, any value in mulhi besides -1 means we need to saturate.
  if (!neg_prod && mulhi != 0)
    return d::max_v<T>();
  if (neg_prod && mulhi != -1)
    return d::max_v<T>(); // essentially some other negative value.
  return __s_add_sat(T(a * b), c);
}

template <typename T> inline T __u_mad_sat(T a, T b, T c) {
  using UPT = typename d::make_upper<T>::type;
  UPT mul = UPT(a) * UPT(b);
  const UPT min = d::min_v<T>();
  const UPT max = d::max_v<T>();
  mul = std::min(std::max(mul, min), max);
  return __u_add_sat(T(mul), c);
}

template <typename T> inline T __u_long_mad_sat(T a, T b, T c) {
  T mulhi = __long_mul_hi(a, b);
  // check mul_hi. If it is any value != 0.
  if (mulhi != 0)
    return d::max_v<T>();
  return __u_add_sat(T(a * b), c);
}

template <typename T> inline T __rotate(T x, T n) {
  using UT = typename std::make_unsigned<T>::type;
  return (x << n) | (UT(x) >> ((sizeof(x) * 8) - n));
}

template <typename T> inline T __u_sub_sat(T x, T y) {
  return (y < (x - d::min_v<T>())) ? (x - y) : d::min_v<T>();
}

template <typename T> inline T __s_sub_sat(T x, T y) {
  if (y > 0)
    return (y < (x - d::min_v<T>()) ? x - y : d::min_v<T>());
  if (y < 0)
    return (y > (x - d::max_v<T>()) ? x - y : d::max_v<T>());
  return x;
}

template <typename T1, typename T2>
typename d::make_upper<T1>::type inline __upsample(T1 hi, T2 lo) {
  using UT = typename d::make_upper<T1>::type;
  return (UT(hi) << (sizeof(T1) * 8)) | lo;
}

template <typename T> inline constexpr T __popcount_impl(T x, size_t n = 0) {
  return (x == T(0)) ? n : __popcount_impl(x >> 1, ((x & T(1)) ? ++n : n));
}

template <typename T> inline constexpr T __popcount(T x) {
  using UT = typename d::make_unsigned<T>::type;
  return __popcount_impl(UT(x));
}

template <typename T> inline T __mad24(T x, T y, T z) { return (x * y) + z; }

template <typename T> inline T __mul24(T x, T y) { return (x * y); }

} // namespace

// --------------- 4.13.4 Integer functions. Host implementations --------------
// u_abs
cl_uchar u_abs(s::cl_uchar x) __NOEXC { return x; }
cl_ushort u_abs(s::cl_ushort x) __NOEXC { return x; }
cl_uint u_abs(s::cl_uint x) __NOEXC { return x; }
cl_ulong u_abs(s::cl_ulong x) __NOEXC { return x; }
MAKE_1V(u_abs, s::cl_uchar, s::cl_uchar)
MAKE_1V(u_abs, s::cl_ushort, s::cl_ushort)
MAKE_1V(u_abs, s::cl_uint, s::cl_uint)
MAKE_1V(u_abs, s::cl_ulong, s::cl_ulong)

// s_abs
cl_uchar s_abs(s::cl_char x) __NOEXC { return std::abs(x); }
cl_ushort s_abs(s::cl_short x) __NOEXC { return std::abs(x); }
cl_uint s_abs(s::cl_int x) __NOEXC { return std::abs(x); }
cl_ulong s_abs(s::cl_long x) __NOEXC { return std::abs(x); }
MAKE_1V(s_abs, s::cl_uchar, s::cl_char)
MAKE_1V(s_abs, s::cl_ushort, s::cl_short)
MAKE_1V(s_abs, s::cl_uint, s::cl_int)
MAKE_1V(s_abs, s::cl_ulong, s::cl_long)

// u_abs_diff
cl_uchar u_abs_diff(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __abs_diff(x, y);
}
cl_ushort u_abs_diff(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __abs_diff(x, y);
}
cl_uint u_abs_diff(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __abs_diff(x, y);
}
cl_ulong u_abs_diff(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __abs_diff(x, y);
}

MAKE_1V_2V(u_abs_diff, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_abs_diff, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_abs_diff, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_abs_diff, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_abs_diff
cl_uchar s_abs_diff(s::cl_char x, s::cl_char y) __NOEXC {
  return __abs_diff(x, y);
}
cl_ushort s_abs_diff(s::cl_short x, s::cl_short y) __NOEXC {
  return __abs_diff(x, y);
}
cl_uint s_abs_diff(s::cl_int x, s::cl_int y) __NOEXC {
  return __abs_diff(x, y);
}
cl_ulong s_abs_diff(s::cl_long x, s::cl_long y) __NOEXC {
  return __abs_diff(x, y);
}
MAKE_1V_2V(s_abs_diff, s::cl_uchar, s::cl_char, s::cl_char)
MAKE_1V_2V(s_abs_diff, s::cl_ushort, s::cl_short, s::cl_short)
MAKE_1V_2V(s_abs_diff, s::cl_uint, s::cl_int, s::cl_int)
MAKE_1V_2V(s_abs_diff, s::cl_ulong, s::cl_long, s::cl_long)

// u_add_sat
cl_uchar u_add_sat(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __u_add_sat(x, y);
}
cl_ushort u_add_sat(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __u_add_sat(x, y);
}
cl_uint u_add_sat(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __u_add_sat(x, y);
}
cl_ulong u_add_sat(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __u_add_sat(x, y);
}
MAKE_1V_2V(u_add_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_add_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_add_sat, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_add_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_add_sat
cl_char s_add_sat(s::cl_char x, s::cl_char y) __NOEXC {
  return __s_add_sat(x, y);
}
cl_short s_add_sat(s::cl_short x, s::cl_short y) __NOEXC {
  return __s_add_sat(x, y);
}
cl_int s_add_sat(s::cl_int x, s::cl_int y) __NOEXC { return __s_add_sat(x, y); }
cl_long s_add_sat(s::cl_long x, s::cl_long y) __NOEXC {
  return __s_add_sat(x, y);
}
MAKE_1V_2V(s_add_sat, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_add_sat, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_add_sat, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_add_sat, s::cl_long, s::cl_long, s::cl_long)

// u_hadd
cl_uchar u_hadd(s::cl_uchar x, s::cl_uchar y) __NOEXC { return __hadd(x, y); }
cl_ushort u_hadd(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __hadd(x, y);
}
cl_uint u_hadd(s::cl_uint x, s::cl_uint y) __NOEXC { return __hadd(x, y); }
cl_ulong u_hadd(s::cl_ulong x, s::cl_ulong y) __NOEXC { return __hadd(x, y); }
MAKE_1V_2V(u_hadd, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_hadd, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_hadd, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_hadd, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_hadd
cl_char s_hadd(s::cl_char x, s::cl_char y) __NOEXC { return __hadd(x, y); }
cl_short s_hadd(s::cl_short x, s::cl_short y) __NOEXC { return __hadd(x, y); }
cl_int s_hadd(s::cl_int x, s::cl_int y) __NOEXC { return __hadd(x, y); }
cl_long s_hadd(s::cl_long x, s::cl_long y) __NOEXC { return __hadd(x, y); }
MAKE_1V_2V(s_hadd, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_hadd, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_hadd, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_hadd, s::cl_long, s::cl_long, s::cl_long)

// u_rhadd
cl_uchar u_rhadd(s::cl_uchar x, s::cl_uchar y) __NOEXC { return __rhadd(x, y); }
cl_ushort u_rhadd(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __rhadd(x, y);
}
cl_uint u_rhadd(s::cl_uint x, s::cl_uint y) __NOEXC { return __rhadd(x, y); }
cl_ulong u_rhadd(s::cl_ulong x, s::cl_ulong y) __NOEXC { return __rhadd(x, y); }
MAKE_1V_2V(u_rhadd, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_rhadd, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_rhadd, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_rhadd, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_rhadd
cl_char s_rhadd(s::cl_char x, s::cl_char y) __NOEXC { return __rhadd(x, y); }
cl_short s_rhadd(s::cl_short x, s::cl_short y) __NOEXC { return __rhadd(x, y); }
cl_int s_rhadd(s::cl_int x, s::cl_int y) __NOEXC { return __rhadd(x, y); }
cl_long s_rhadd(s::cl_long x, s::cl_long y) __NOEXC { return __rhadd(x, y); }
MAKE_1V_2V(s_rhadd, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_rhadd, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_rhadd, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_rhadd, s::cl_long, s::cl_long, s::cl_long)

// u_clamp
cl_uchar u_clamp(s::cl_uchar x, s::cl_uchar minval,
                 s::cl_uchar maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_ushort u_clamp(s::cl_ushort x, s::cl_ushort minval,
                  s::cl_ushort maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_uint u_clamp(s::cl_uint x, s::cl_uint minval, s::cl_uint maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_ulong u_clamp(s::cl_ulong x, s::cl_ulong minval,
                 s::cl_ulong maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
MAKE_1V_2V_3V(u_clamp, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V_3V(u_clamp, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V_3V(u_clamp, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V_3V(u_clamp, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S_3S(u_clamp, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S_3S(u_clamp, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S_3S(u_clamp, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S_3S(u_clamp, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_clamp
cl_char s_clamp(s::cl_char x, s::cl_char minval, s::cl_char maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_short s_clamp(s::cl_short x, s::cl_short minval,
                 s::cl_short maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_int s_clamp(s::cl_int x, s::cl_int minval, s::cl_int maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_long s_clamp(s::cl_long x, s::cl_long minval, s::cl_long maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
MAKE_1V_2V_3V(s_clamp, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V_3V(s_clamp, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V_3V(s_clamp, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V_3V(s_clamp, s::cl_long, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S_3S(s_clamp, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S_3S(s_clamp, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S_3S(s_clamp, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S_3S(s_clamp, s::cl_long, s::cl_long, s::cl_long, s::cl_long)

// clz
cl_uchar clz(s::cl_uchar x) __NOEXC { return __clz(x); }
cl_char clz(s::cl_char x) __NOEXC { return __clz(x); }
cl_ushort clz(s::cl_ushort x) __NOEXC { return __clz(x); }
cl_short clz(s::cl_short x) __NOEXC { return __clz(x); }
cl_uint clz(s::cl_uint x) __NOEXC { return __clz(x); }
cl_int clz(s::cl_int x) __NOEXC { return __clz(x); }
cl_ulong clz(s::cl_ulong x) __NOEXC { return __clz(x); }
cl_long clz(s::cl_long x) __NOEXC { return __clz(x); }
MAKE_1V(clz, s::cl_uchar, s::cl_uchar)
MAKE_1V(clz, s::cl_char, s::cl_char)
MAKE_1V(clz, s::cl_ushort, s::cl_ushort)
MAKE_1V(clz, s::cl_short, s::cl_short)
MAKE_1V(clz, s::cl_uint, s::cl_uint)
MAKE_1V(clz, s::cl_int, s::cl_int)
MAKE_1V(clz, s::cl_ulong, s::cl_ulong)
MAKE_1V(clz, s::cl_long, s::cl_long)

// s_mul_hi
cl_char s_mul_hi(cl_char a, cl_char b) { return __mul_hi(a, b); }
cl_short s_mul_hi(cl_short a, cl_short b) { return __mul_hi(a, b); }
cl_int s_mul_hi(cl_int a, cl_int b) { return __mul_hi(a, b); }
cl_long s_mul_hi(s::cl_long x, s::cl_long y) __NOEXC {
  return __long_mul_hi(x, y);
}
MAKE_1V_2V(s_mul_hi, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_mul_hi, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_mul_hi, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_mul_hi, s::cl_long, s::cl_long, s::cl_long)

// u_mul_hi
cl_uchar u_mul_hi(cl_uchar a, cl_uchar b) { return __mul_hi(a, b); }
cl_ushort u_mul_hi(cl_ushort a, cl_ushort b) { return __mul_hi(a, b); }
cl_uint u_mul_hi(cl_uint a, cl_uint b) { return __mul_hi(a, b); }
cl_ulong u_mul_hi(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __long_mul_hi(x, y);
}
MAKE_1V_2V(u_mul_hi, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_mul_hi, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_mul_hi, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_mul_hi, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_mad_hi
cl_char s_mad_hi(s::cl_char x, s::cl_char minval, s::cl_char maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_short s_mad_hi(s::cl_short x, s::cl_short minval,
                  s::cl_short maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_int s_mad_hi(s::cl_int x, s::cl_int minval, s::cl_int maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_long s_mad_hi(s::cl_long x, s::cl_long minval, s::cl_long maxval) __NOEXC {
  return __long_mad_hi(x, minval, maxval);
}
MAKE_1V_2V_3V(s_mad_hi, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V_3V(s_mad_hi, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V_3V(s_mad_hi, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V_3V(s_mad_hi, s::cl_long, s::cl_long, s::cl_long, s::cl_long)

// u_mad_hi
cl_uchar u_mad_hi(s::cl_uchar x, s::cl_uchar minval,
                  s::cl_uchar maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_ushort u_mad_hi(s::cl_ushort x, s::cl_ushort minval,
                   s::cl_ushort maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_uint u_mad_hi(s::cl_uint x, s::cl_uint minval, s::cl_uint maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_ulong u_mad_hi(s::cl_ulong x, s::cl_ulong minval,
                  s::cl_ulong maxval) __NOEXC {
  return __long_mad_hi(x, minval, maxval);
}
MAKE_1V_2V_3V(u_mad_hi, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V_3V(u_mad_hi, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V_3V(u_mad_hi, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V_3V(u_mad_hi, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_mad_sat
cl_char s_mad_sat(s::cl_char a, s::cl_char b, s::cl_char c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
cl_short s_mad_sat(s::cl_short a, s::cl_short b, s::cl_short c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
cl_int s_mad_sat(s::cl_int a, s::cl_int b, s::cl_int c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
cl_long s_mad_sat(s::cl_long a, s::cl_long b, s::cl_long c) __NOEXC {
  return __s_long_mad_sat(a, b, c);
}
MAKE_1V_2V_3V(s_mad_sat, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V_3V(s_mad_sat, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V_3V(s_mad_sat, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V_3V(s_mad_sat, s::cl_long, s::cl_long, s::cl_long, s::cl_long)

// u_mad_sat
cl_uchar u_mad_sat(s::cl_uchar a, s::cl_uchar b, s::cl_uchar c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
cl_ushort u_mad_sat(s::cl_ushort a, s::cl_ushort b, s::cl_ushort c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
cl_uint u_mad_sat(s::cl_uint a, s::cl_uint b, s::cl_uint c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
cl_ulong u_mad_sat(s::cl_ulong a, s::cl_ulong b, s::cl_ulong c) __NOEXC {
  return __u_long_mad_sat(a, b, c);
}
MAKE_1V_2V_3V(u_mad_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V_3V(u_mad_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V_3V(u_mad_sat, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V_3V(u_mad_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_max
cl_char s_max(s::cl_char x, s::cl_char y) __NOEXC { return std::max(x, y); }
cl_short s_max(s::cl_short x, s::cl_short y) __NOEXC { return std::max(x, y); }
cl_int s_max(s::cl_int x, s::cl_int y) __NOEXC { return std::max(x, y); }
cl_long s_max(s::cl_long x, s::cl_long y) __NOEXC { return std::max(x, y); }
MAKE_1V_2V(s_max, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_max, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_max, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_max, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S(s_max, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S(s_max, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S(s_max, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S(s_max, s::cl_long, s::cl_long, s::cl_long)

// u_max
cl_uchar u_max(s::cl_uchar x, s::cl_uchar y) __NOEXC { return std::max(x, y); }
cl_ushort u_max(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return std::max(x, y);
}
cl_uint u_max(s::cl_uint x, s::cl_uint y) __NOEXC { return std::max(x, y); }
cl_ulong u_max(s::cl_ulong x, s::cl_ulong y) __NOEXC { return std::max(x, y); }
MAKE_1V_2V(u_max, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_max, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_max, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_max, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S(u_max, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S(u_max, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S(u_max, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S(u_max, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_min
cl_char s_min(s::cl_char x, s::cl_char y) __NOEXC { return std::min(x, y); }
cl_short s_min(s::cl_short x, s::cl_short y) __NOEXC { return std::min(x, y); }
cl_int s_min(s::cl_int x, s::cl_int y) __NOEXC { return std::min(x, y); }
cl_long s_min(s::cl_long x, s::cl_long y) __NOEXC { return std::min(x, y); }
MAKE_1V_2V(s_min, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_min, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_min, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_min, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S(s_min, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S(s_min, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S(s_min, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S(s_min, s::cl_long, s::cl_long, s::cl_long)

// u_min
cl_uchar u_min(s::cl_uchar x, s::cl_uchar y) __NOEXC { return std::min(x, y); }
cl_ushort u_min(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return std::min(x, y);
}
cl_uint u_min(s::cl_uint x, s::cl_uint y) __NOEXC { return std::min(x, y); }
cl_ulong u_min(s::cl_ulong x, s::cl_ulong y) __NOEXC { return std::min(x, y); }
MAKE_1V_2V(u_min, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_min, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_min, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_min, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S(u_min, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S(u_min, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S(u_min, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S(u_min, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// rotate
cl_uchar rotate(s::cl_uchar x, s::cl_uchar y) __NOEXC { return __rotate(x, y); }
cl_ushort rotate(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __rotate(x, y);
}
cl_uint rotate(s::cl_uint x, s::cl_uint y) __NOEXC { return __rotate(x, y); }
cl_ulong rotate(s::cl_ulong x, s::cl_ulong y) __NOEXC { return __rotate(x, y); }
cl_char rotate(s::cl_char x, s::cl_char y) __NOEXC { return __rotate(x, y); }
cl_short rotate(s::cl_short x, s::cl_short y) __NOEXC { return __rotate(x, y); }
cl_int rotate(s::cl_int x, s::cl_int y) __NOEXC { return __rotate(x, y); }
cl_long rotate(s::cl_long x, s::cl_long y) __NOEXC { return __rotate(x, y); }
MAKE_1V_2V(rotate, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(rotate, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(rotate, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(rotate, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(rotate, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(rotate, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(rotate, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(rotate, s::cl_long, s::cl_long, s::cl_long)

// u_sub_sat
cl_uchar u_sub_sat(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __u_sub_sat(x, y);
}
cl_ushort u_sub_sat(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __u_sub_sat(x, y);
}
cl_uint u_sub_sat(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __u_sub_sat(x, y);
}
cl_ulong u_sub_sat(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __u_sub_sat(x, y);
}
MAKE_1V_2V(u_sub_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_sub_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_sub_sat, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_sub_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_sub_sat
cl_char s_sub_sat(s::cl_char x, s::cl_char y) __NOEXC {
  return __s_sub_sat(x, y);
}
cl_short s_sub_sat(s::cl_short x, s::cl_short y) __NOEXC {
  return __s_sub_sat(x, y);
}
cl_int s_sub_sat(s::cl_int x, s::cl_int y) __NOEXC { return __s_sub_sat(x, y); }
cl_long s_sub_sat(s::cl_long x, s::cl_long y) __NOEXC {
  return __s_sub_sat(x, y);
}
MAKE_1V_2V(s_sub_sat, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_sub_sat, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_sub_sat, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_sub_sat, s::cl_long, s::cl_long, s::cl_long)

// u_upsample
cl_ushort u_upsample(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __upsample(x, y);
}
cl_uint u_upsample(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __upsample(x, y);
}
cl_ulong u_upsample(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __upsample(x, y);
}
MAKE_1V_2V(u_upsample, s::cl_ushort, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_upsample, s::cl_uint, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_upsample, s::cl_ulong, s::cl_uint, s::cl_uint)

// TODO delete when Intel CPU OpenCL runtime will be fixed
// ExtInst ... s_upsample -> _Z8upsampleij (now _Z8upsampleii)
#define s_upsample u_upsample

cl_short s_upsample(s::cl_char x, s::cl_uchar y) __NOEXC {
  return __upsample(x, y);
}
cl_int s_upsample(s::cl_short x, s::cl_ushort y) __NOEXC {
  return __upsample(x, y);
}
cl_long s_upsample(s::cl_int x, s::cl_uint y) __NOEXC {
  return __upsample(x, y);
}
MAKE_1V_2V(s_upsample, s::cl_short, s::cl_char, s::cl_uchar)
MAKE_1V_2V(s_upsample, s::cl_int, s::cl_short, s::cl_ushort)
MAKE_1V_2V(s_upsample, s::cl_long, s::cl_int, s::cl_uint)

#undef s_upsample

// popcount
cl_uchar popcount(s::cl_uchar x) __NOEXC { return __popcount(x); }
cl_ushort popcount(s::cl_ushort x) __NOEXC { return __popcount(x); }
cl_uint popcount(s::cl_uint x) __NOEXC { return __popcount(x); }
cl_ulong popcount(s::cl_ulong x) __NOEXC { return __popcount(x); }
MAKE_1V(popcount, s::cl_uchar, s::cl_uchar)
MAKE_1V(popcount, s::cl_ushort, s::cl_ushort)
MAKE_1V(popcount, s::cl_uint, s::cl_uint)
MAKE_1V(popcount, s::cl_ulong, s::cl_ulong)

cl_char popcount(s::cl_char x) __NOEXC { return __popcount(x); }
cl_short popcount(s::cl_short x) __NOEXC { return __popcount(x); }
cl_int popcount(s::cl_int x) __NOEXC { return __popcount(x); }
cl_long popcount(s::cl_long x) __NOEXC { return __popcount(x); }
MAKE_1V(popcount, s::cl_char, s::cl_char)
MAKE_1V(popcount, s::cl_short, s::cl_short)
MAKE_1V(popcount, s::cl_int, s::cl_int)
MAKE_1V(popcount, s::cl_long, s::cl_long)

// u_mad24
cl_uint u_mad24(s::cl_uint x, s::cl_uint y, s::cl_uint z) __NOEXC {
  return __mad24(x, y, z);
}
MAKE_1V_2V_3V(u_mad24, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)

// s_mad24
cl_int s_mad24(s::cl_int x, s::cl_int y, s::cl_int z) __NOEXC {
  return __mad24(x, y, z);
}
MAKE_1V_2V_3V(s_mad24, s::cl_int, s::cl_int, s::cl_int, s::cl_int)

// u_mul24
cl_uint u_mul24(s::cl_uint x, s::cl_uint y) __NOEXC { return __mul24(x, y); }
MAKE_1V_2V(u_mul24, s::cl_uint, s::cl_uint, s::cl_uint)

// s_mul24
cl_int s_mul24(s::cl_int x, s::cl_int y) __NOEXC { return __mul24(x, y); }
MAKE_1V_2V(s_mul24, s::cl_int, s::cl_int, s::cl_int)

} // namespace __host_std
} // namespace cl
