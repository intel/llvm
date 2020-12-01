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
#include <CL/sycl/detail/export.hpp>

#include <algorithm>
#include <type_traits>

namespace s = cl::sycl;
namespace d = s::detail;

__SYCL_INLINE_NAMESPACE(cl) {
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

template <typename T> inline constexpr T __ctz_impl(T x, T m, T n = 0) {
  return (x & m) ? n : __ctz_impl(x, T(m << 1), ++n);
}

template <typename T> inline constexpr T __ctz(T x) {
  using UT = typename std::make_unsigned<T>::type;
  return (x == T(0)) ? sizeof(T) * 8 : __ctz_impl<UT>(x, 1);
}

template <typename T> T __mul_hi(T a, T b) {
  using UPT = typename d::make_larger<T>::type;
  UPT a_s = a;
  UPT b_s = b;
  UPT mul = a_s * b_s;
  return (mul >> (sizeof(T) * 8));
}

// A helper function for mul_hi built-in for long
template <typename T> inline T __get_high_half(T a0b0, T a0b1, T a1b0, T a1b1) {
  constexpr int halfsize = (sizeof(T) * 8) / 2;
  // To get the upper 64 bits:
  // 64 bits from a1b1, upper 32 bits from [a1b0 + (a0b1 + a0b0>>32 (carry bit
  // in 33rd bit))] with carry bit on 64th bit - use of hadd. Add the a1b1 to
  // the above 32 bit result.
  return a1b1 + (__hadd(a1b0, (a0b1 + (a0b0 >> halfsize))) >> (halfsize - 1));
}

// A helper function for mul_hi built-in for long
template <typename T>
inline void __get_half_products(T a, T b, T &a0b0, T &a0b1, T &a1b0, T &a1b1) {
  constexpr s::cl_int halfsize = (sizeof(T) * 8) / 2;
  T a1 = a >> halfsize;
  T a0 = (a << halfsize) >> halfsize;
  T b1 = b >> halfsize;
  T b0 = (b << halfsize) >> halfsize;

  // a1b1 - for bits - [64-128)
  // a1b0 a0b1 for bits - [32-96)
  // a0b0 for bits - [0-64)
  a1b1 = a1 * b1;
  a0b1 = a0 * b1;
  a1b0 = a1 * b0;
  a0b0 = a0 * b0;
}

// T is minimum of 64 bits- long or longlong
template <typename T> inline T __u_long_mul_hi(T a, T b) {
  T a0b0, a0b1, a1b0, a1b1;
  __get_half_products(a, b, a0b0, a0b1, a1b0, a1b1);
  T result = __get_high_half(a0b0, a0b1, a1b0, a1b1);
  return result;
}

template <typename T> inline T __s_long_mul_hi(T a, T b) {
  using UT = typename std::make_unsigned<T>::type;
  UT absA = std::abs(a);
  UT absB = std::abs(b);

  UT a0b0, a0b1, a1b0, a1b1;
  __get_half_products(absA, absB, a0b0, a0b1, a1b0, a1b1);
  T result = __get_high_half(a0b0, a0b1, a1b0, a1b1);

  bool isResultNegative = (a < 0) != (b < 0);
  if (isResultNegative) {
    result = ~result;

    // Find the low half to see if we need to carry
    constexpr int halfsize = (sizeof(T) * 8) / 2;
    UT low = a0b0 + ((a0b1 + a1b0) << halfsize);
    if (low == 0)
      ++result;
  }

  return result;
}

template <typename T> inline T __mad_hi(T a, T b, T c) {
  return __mul_hi(a, b) + c;
}

template <typename T> inline T __u_long_mad_hi(T a, T b, T c) {
  return __u_long_mul_hi(a, b) + c;
}

template <typename T> inline T __s_long_mad_hi(T a, T b, T c) {
  return __s_long_mul_hi(a, b) + c;
}

template <typename T> inline T __s_mad_sat(T a, T b, T c) {
  using UPT = typename d::make_larger<T>::type;
  UPT mul = UPT(a) * UPT(b);
  UPT res = mul + UPT(c);
  const UPT max = d::max_v<T>();
  const UPT min = d::min_v<T>();
  res = std::min(std::max(res, min), max);
  return T(res);
}

template <typename T> inline T __s_long_mad_sat(T a, T b, T c) {
  bool neg_prod = (a < 0) ^ (b < 0);
  T mulhi = __s_long_mul_hi(a, b);

  // check mul_hi. If it is any value != 0.
  // if prod is +ve, any value in mulhi means we need to saturate.
  // if prod is -ve, any value in mulhi besides -1 means we need to saturate.
  if (!neg_prod && mulhi != 0)
    return d::max_v<T>();
  if (neg_prod && mulhi != -1)
    return d::min_v<T>(); // essentially some other negative value.
  return __s_add_sat(T(a * b), c);
}

template <typename T> inline T __u_mad_sat(T a, T b, T c) {
  using UPT = typename d::make_larger<T>::type;
  UPT mul = UPT(a) * UPT(b);
  const UPT min = d::min_v<T>();
  const UPT max = d::max_v<T>();
  mul = std::min(std::max(mul, min), max);
  return __u_add_sat(T(mul), c);
}

template <typename T> inline T __u_long_mad_sat(T a, T b, T c) {
  T mulhi = __u_long_mul_hi(a, b);
  // check mul_hi. If it is any value != 0.
  if (mulhi != 0)
    return d::max_v<T>();
  return __u_add_sat(T(a * b), c);
}

template <typename T> inline T __rotate(T x, T n) {
  using UT = typename std::make_unsigned<T>::type;
  // Shrink the shift width so that it's in the range [0, num_bits(T)). Cast
  // everything to unsigned to avoid type conversion issues.
  constexpr UT size = sizeof(x) * 8;
  UT xu = UT(x);
  UT nu = UT(n) & (size - 1);
  return (xu << nu) | (xu >> (size - nu));
}

template <typename T> inline T __u_sub_sat(T x, T y) {
  return (y < (x - d::min_v<T>())) ? (x - y) : d::min_v<T>();
}

template <typename T> inline T __s_sub_sat(T x, T y) {
  using UT = typename std::make_unsigned<T>::type;
  T result = UT(x) - UT(y);
  // Saturate result if (+) - (-) = (-) or (-) - (+) = (+).
  if (((x < 0) ^ (y < 0)) && ((x < 0) ^ (result < 0)))
    result = result < 0 ? d::max_v<T>() : d::min_v<T>();
  return result;
}

template <typename T1, typename T2>
typename d::make_larger<T1>::type inline __upsample(T1 hi, T2 lo) {
  using UT = typename d::make_larger<T1>::type;
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
__SYCL_EXPORT s::cl_uchar u_abs(s::cl_uchar x) __NOEXC { return x; }
__SYCL_EXPORT s::cl_ushort u_abs(s::cl_ushort x) __NOEXC { return x; }
__SYCL_EXPORT s::cl_uint u_abs(s::cl_uint x) __NOEXC { return x; }
__SYCL_EXPORT s::cl_ulong u_abs(s::cl_ulong x) __NOEXC { return x; }
MAKE_1V(u_abs, s::cl_uchar, s::cl_uchar)
MAKE_1V(u_abs, s::cl_ushort, s::cl_ushort)
MAKE_1V(u_abs, s::cl_uint, s::cl_uint)
MAKE_1V(u_abs, s::cl_ulong, s::cl_ulong)

// s_abs
__SYCL_EXPORT s::cl_uchar s_abs(s::cl_char x) __NOEXC { return std::abs(x); }
__SYCL_EXPORT s::cl_ushort s_abs(s::cl_short x) __NOEXC { return std::abs(x); }
__SYCL_EXPORT s::cl_uint s_abs(s::cl_int x) __NOEXC { return std::abs(x); }
__SYCL_EXPORT s::cl_ulong s_abs(s::cl_long x) __NOEXC { return std::abs(x); }
MAKE_1V(s_abs, s::cl_uchar, s::cl_char)
MAKE_1V(s_abs, s::cl_ushort, s::cl_short)
MAKE_1V(s_abs, s::cl_uint, s::cl_int)
MAKE_1V(s_abs, s::cl_ulong, s::cl_long)

// u_abs_diff
__SYCL_EXPORT s::cl_uchar u_abs_diff(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __abs_diff(x, y);
}
__SYCL_EXPORT s::cl_ushort u_abs_diff(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __abs_diff(x, y);
}
__SYCL_EXPORT s::cl_uint u_abs_diff(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __abs_diff(x, y);
}
__SYCL_EXPORT s::cl_ulong u_abs_diff(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __abs_diff(x, y);
}

MAKE_1V_2V(u_abs_diff, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_abs_diff, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_abs_diff, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_abs_diff, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_abs_diff
__SYCL_EXPORT s::cl_uchar s_abs_diff(s::cl_char x, s::cl_char y) __NOEXC {
  return __abs_diff(x, y);
}
__SYCL_EXPORT s::cl_ushort s_abs_diff(s::cl_short x, s::cl_short y) __NOEXC {
  return __abs_diff(x, y);
}
__SYCL_EXPORT s::cl_uint s_abs_diff(s::cl_int x, s::cl_int y) __NOEXC {
  return __abs_diff(x, y);
}
__SYCL_EXPORT s::cl_ulong s_abs_diff(s::cl_long x, s::cl_long y) __NOEXC {
  return __abs_diff(x, y);
}
MAKE_1V_2V(s_abs_diff, s::cl_uchar, s::cl_char, s::cl_char)
MAKE_1V_2V(s_abs_diff, s::cl_ushort, s::cl_short, s::cl_short)
MAKE_1V_2V(s_abs_diff, s::cl_uint, s::cl_int, s::cl_int)
MAKE_1V_2V(s_abs_diff, s::cl_ulong, s::cl_long, s::cl_long)

// u_add_sat
__SYCL_EXPORT s::cl_uchar u_add_sat(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __u_add_sat(x, y);
}
__SYCL_EXPORT s::cl_ushort u_add_sat(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __u_add_sat(x, y);
}
__SYCL_EXPORT s::cl_uint u_add_sat(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __u_add_sat(x, y);
}
__SYCL_EXPORT s::cl_ulong u_add_sat(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __u_add_sat(x, y);
}
MAKE_1V_2V(u_add_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_add_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_add_sat, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_add_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_add_sat
__SYCL_EXPORT s::cl_char s_add_sat(s::cl_char x, s::cl_char y) __NOEXC {
  return __s_add_sat(x, y);
}
__SYCL_EXPORT s::cl_short s_add_sat(s::cl_short x, s::cl_short y) __NOEXC {
  return __s_add_sat(x, y);
}
__SYCL_EXPORT s::cl_int s_add_sat(s::cl_int x, s::cl_int y) __NOEXC {
  return __s_add_sat(x, y);
}
__SYCL_EXPORT s::cl_long s_add_sat(s::cl_long x, s::cl_long y) __NOEXC {
  return __s_add_sat(x, y);
}
MAKE_1V_2V(s_add_sat, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_add_sat, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_add_sat, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_add_sat, s::cl_long, s::cl_long, s::cl_long)

// u_hadd
__SYCL_EXPORT s::cl_uchar u_hadd(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __hadd(x, y);
}
__SYCL_EXPORT s::cl_ushort u_hadd(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __hadd(x, y);
}
__SYCL_EXPORT s::cl_uint u_hadd(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __hadd(x, y);
}
__SYCL_EXPORT s::cl_ulong u_hadd(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __hadd(x, y);
}
MAKE_1V_2V(u_hadd, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_hadd, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_hadd, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_hadd, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_hadd
__SYCL_EXPORT s::cl_char s_hadd(s::cl_char x, s::cl_char y) __NOEXC {
  return __hadd(x, y);
}
__SYCL_EXPORT s::cl_short s_hadd(s::cl_short x, s::cl_short y) __NOEXC {
  return __hadd(x, y);
}
__SYCL_EXPORT s::cl_int s_hadd(s::cl_int x, s::cl_int y) __NOEXC {
  return __hadd(x, y);
}
__SYCL_EXPORT s::cl_long s_hadd(s::cl_long x, s::cl_long y) __NOEXC {
  return __hadd(x, y);
}
MAKE_1V_2V(s_hadd, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_hadd, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_hadd, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_hadd, s::cl_long, s::cl_long, s::cl_long)

// u_rhadd
__SYCL_EXPORT s::cl_uchar u_rhadd(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __rhadd(x, y);
}
__SYCL_EXPORT s::cl_ushort u_rhadd(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __rhadd(x, y);
}
__SYCL_EXPORT s::cl_uint u_rhadd(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __rhadd(x, y);
}
__SYCL_EXPORT s::cl_ulong u_rhadd(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __rhadd(x, y);
}
MAKE_1V_2V(u_rhadd, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_rhadd, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_rhadd, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_rhadd, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_rhadd
__SYCL_EXPORT s::cl_char s_rhadd(s::cl_char x, s::cl_char y) __NOEXC {
  return __rhadd(x, y);
}
__SYCL_EXPORT s::cl_short s_rhadd(s::cl_short x, s::cl_short y) __NOEXC {
  return __rhadd(x, y);
}
__SYCL_EXPORT s::cl_int s_rhadd(s::cl_int x, s::cl_int y) __NOEXC {
  return __rhadd(x, y);
}
__SYCL_EXPORT s::cl_long s_rhadd(s::cl_long x, s::cl_long y) __NOEXC {
  return __rhadd(x, y);
}
MAKE_1V_2V(s_rhadd, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_rhadd, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_rhadd, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_rhadd, s::cl_long, s::cl_long, s::cl_long)

// u_clamp
__SYCL_EXPORT s::cl_uchar u_clamp(s::cl_uchar x, s::cl_uchar minval,
                                  s::cl_uchar maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
__SYCL_EXPORT s::cl_ushort u_clamp(s::cl_ushort x, s::cl_ushort minval,
                                   s::cl_ushort maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
__SYCL_EXPORT s::cl_uint u_clamp(s::cl_uint x, s::cl_uint minval,
                                 s::cl_uint maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
__SYCL_EXPORT s::cl_ulong u_clamp(s::cl_ulong x, s::cl_ulong minval,
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
__SYCL_EXPORT s::cl_char s_clamp(s::cl_char x, s::cl_char minval,
                                 s::cl_char maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
__SYCL_EXPORT s::cl_short s_clamp(s::cl_short x, s::cl_short minval,
                                  s::cl_short maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
__SYCL_EXPORT s::cl_int s_clamp(s::cl_int x, s::cl_int minval,
                                s::cl_int maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
__SYCL_EXPORT s::cl_long s_clamp(s::cl_long x, s::cl_long minval,
                                 s::cl_long maxval) __NOEXC {
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
__SYCL_EXPORT s::cl_uchar clz(s::cl_uchar x) __NOEXC { return __clz(x); }
__SYCL_EXPORT s::cl_char clz(s::cl_char x) __NOEXC { return __clz(x); }
__SYCL_EXPORT s::cl_ushort clz(s::cl_ushort x) __NOEXC { return __clz(x); }
__SYCL_EXPORT s::cl_short clz(s::cl_short x) __NOEXC { return __clz(x); }
__SYCL_EXPORT s::cl_uint clz(s::cl_uint x) __NOEXC { return __clz(x); }
__SYCL_EXPORT s::cl_int clz(s::cl_int x) __NOEXC { return __clz(x); }
__SYCL_EXPORT s::cl_ulong clz(s::cl_ulong x) __NOEXC { return __clz(x); }
__SYCL_EXPORT s::cl_long clz(s::cl_long x) __NOEXC { return __clz(x); }
MAKE_1V(clz, s::cl_uchar, s::cl_uchar)
MAKE_1V(clz, s::cl_char, s::cl_char)
MAKE_1V(clz, s::cl_ushort, s::cl_ushort)
MAKE_1V(clz, s::cl_short, s::cl_short)
MAKE_1V(clz, s::cl_uint, s::cl_uint)
MAKE_1V(clz, s::cl_int, s::cl_int)
MAKE_1V(clz, s::cl_ulong, s::cl_ulong)
MAKE_1V(clz, s::cl_long, s::cl_long)

// ctz
__SYCL_EXPORT s::cl_uchar ctz(s::cl_uchar x) __NOEXC { return __ctz(x); }
__SYCL_EXPORT s::cl_char ctz(s::cl_char x) __NOEXC { return __ctz(x); }
__SYCL_EXPORT s::cl_ushort ctz(s::cl_ushort x) __NOEXC { return __ctz(x); }
__SYCL_EXPORT s::cl_short ctz(s::cl_short x) __NOEXC { return __ctz(x); }
__SYCL_EXPORT s::cl_uint ctz(s::cl_uint x) __NOEXC { return __ctz(x); }
__SYCL_EXPORT s::cl_int ctz(s::cl_int x) __NOEXC { return __ctz(x); }
__SYCL_EXPORT s::cl_ulong ctz(s::cl_ulong x) __NOEXC { return __ctz(x); }
__SYCL_EXPORT s::cl_long ctz(s::cl_long x) __NOEXC { return __ctz(x); }
MAKE_1V(ctz, s::cl_uchar, s::cl_uchar)
MAKE_1V(ctz, s::cl_char, s::cl_char)
MAKE_1V(ctz, s::cl_ushort, s::cl_ushort)
MAKE_1V(ctz, s::cl_short, s::cl_short)
MAKE_1V(ctz, s::cl_uint, s::cl_uint)
MAKE_1V(ctz, s::cl_int, s::cl_int)
MAKE_1V(ctz, s::cl_ulong, s::cl_ulong)
MAKE_1V(ctz, s::cl_long, s::cl_long)

// s_mul_hi
__SYCL_EXPORT s::cl_char s_mul_hi(s::cl_char a, s::cl_char b) {
  return __mul_hi(a, b);
}
__SYCL_EXPORT s::cl_short s_mul_hi(s::cl_short a, s::cl_short b) {
  return __mul_hi(a, b);
}
__SYCL_EXPORT s::cl_int s_mul_hi(s::cl_int a, s::cl_int b) {
  return __mul_hi(a, b);
}
__SYCL_EXPORT s::cl_long s_mul_hi(s::cl_long x, s::cl_long y) __NOEXC {
  return __s_long_mul_hi(x, y);
}
MAKE_1V_2V(s_mul_hi, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_mul_hi, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_mul_hi, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_mul_hi, s::cl_long, s::cl_long, s::cl_long)

// u_mul_hi
__SYCL_EXPORT s::cl_uchar u_mul_hi(s::cl_uchar a, s::cl_uchar b) {
  return __mul_hi(a, b);
}
__SYCL_EXPORT s::cl_ushort u_mul_hi(s::cl_ushort a, s::cl_ushort b) {
  return __mul_hi(a, b);
}
__SYCL_EXPORT s::cl_uint u_mul_hi(s::cl_uint a, s::cl_uint b) {
  return __mul_hi(a, b);
}
__SYCL_EXPORT s::cl_ulong u_mul_hi(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __u_long_mul_hi(x, y);
}
MAKE_1V_2V(u_mul_hi, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_mul_hi, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_mul_hi, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_mul_hi, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_mad_hi
__SYCL_EXPORT s::cl_char s_mad_hi(s::cl_char x, s::cl_char minval,
                                  s::cl_char maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
__SYCL_EXPORT s::cl_short s_mad_hi(s::cl_short x, s::cl_short minval,
                                   s::cl_short maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
__SYCL_EXPORT s::cl_int s_mad_hi(s::cl_int x, s::cl_int minval,
                                 s::cl_int maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
__SYCL_EXPORT s::cl_long s_mad_hi(s::cl_long x, s::cl_long minval,
                                  s::cl_long maxval) __NOEXC {
  return __s_long_mad_hi(x, minval, maxval);
}
MAKE_1V_2V_3V(s_mad_hi, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V_3V(s_mad_hi, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V_3V(s_mad_hi, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V_3V(s_mad_hi, s::cl_long, s::cl_long, s::cl_long, s::cl_long)

// u_mad_hi
__SYCL_EXPORT s::cl_uchar u_mad_hi(s::cl_uchar x, s::cl_uchar minval,
                                   s::cl_uchar maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
__SYCL_EXPORT s::cl_ushort u_mad_hi(s::cl_ushort x, s::cl_ushort minval,
                                    s::cl_ushort maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
__SYCL_EXPORT s::cl_uint u_mad_hi(s::cl_uint x, s::cl_uint minval,
                                  s::cl_uint maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
__SYCL_EXPORT s::cl_ulong u_mad_hi(s::cl_ulong x, s::cl_ulong minval,
                                   s::cl_ulong maxval) __NOEXC {
  return __u_long_mad_hi(x, minval, maxval);
}
MAKE_1V_2V_3V(u_mad_hi, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V_3V(u_mad_hi, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V_3V(u_mad_hi, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V_3V(u_mad_hi, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_mad_sat
__SYCL_EXPORT s::cl_char s_mad_sat(s::cl_char a, s::cl_char b,
                                   s::cl_char c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
__SYCL_EXPORT s::cl_short s_mad_sat(s::cl_short a, s::cl_short b,
                                    s::cl_short c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
__SYCL_EXPORT s::cl_int s_mad_sat(s::cl_int a, s::cl_int b,
                                  s::cl_int c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
__SYCL_EXPORT s::cl_long s_mad_sat(s::cl_long a, s::cl_long b,
                                   s::cl_long c) __NOEXC {
  return __s_long_mad_sat(a, b, c);
}
MAKE_1V_2V_3V(s_mad_sat, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V_3V(s_mad_sat, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V_3V(s_mad_sat, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V_3V(s_mad_sat, s::cl_long, s::cl_long, s::cl_long, s::cl_long)

// u_mad_sat
__SYCL_EXPORT s::cl_uchar u_mad_sat(s::cl_uchar a, s::cl_uchar b,
                                    s::cl_uchar c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
__SYCL_EXPORT s::cl_ushort u_mad_sat(s::cl_ushort a, s::cl_ushort b,
                                     s::cl_ushort c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
__SYCL_EXPORT s::cl_uint u_mad_sat(s::cl_uint a, s::cl_uint b,
                                   s::cl_uint c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
__SYCL_EXPORT s::cl_ulong u_mad_sat(s::cl_ulong a, s::cl_ulong b,
                                    s::cl_ulong c) __NOEXC {
  return __u_long_mad_sat(a, b, c);
}
MAKE_1V_2V_3V(u_mad_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V_3V(u_mad_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V_3V(u_mad_sat, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V_3V(u_mad_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_max
__SYCL_EXPORT s::cl_char s_max(s::cl_char x, s::cl_char y) __NOEXC {
  return std::max(x, y);
}
__SYCL_EXPORT s::cl_short s_max(s::cl_short x, s::cl_short y) __NOEXC {
  return std::max(x, y);
}
__SYCL_EXPORT s::cl_int s_max(s::cl_int x, s::cl_int y) __NOEXC {
  return std::max(x, y);
}
__SYCL_EXPORT s::cl_long s_max(s::cl_long x, s::cl_long y) __NOEXC {
  return std::max(x, y);
}
MAKE_1V_2V(s_max, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_max, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_max, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_max, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S(s_max, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S(s_max, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S(s_max, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S(s_max, s::cl_long, s::cl_long, s::cl_long)

// u_max
__SYCL_EXPORT s::cl_uchar u_max(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return std::max(x, y);
}
__SYCL_EXPORT s::cl_ushort u_max(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return std::max(x, y);
}
__SYCL_EXPORT s::cl_uint u_max(s::cl_uint x, s::cl_uint y) __NOEXC {
  return std::max(x, y);
}
__SYCL_EXPORT s::cl_ulong u_max(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return std::max(x, y);
}
MAKE_1V_2V(u_max, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_max, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_max, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_max, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S(u_max, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S(u_max, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S(u_max, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S(u_max, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_min
__SYCL_EXPORT s::cl_char s_min(s::cl_char x, s::cl_char y) __NOEXC {
  return std::min(x, y);
}
__SYCL_EXPORT s::cl_short s_min(s::cl_short x, s::cl_short y) __NOEXC {
  return std::min(x, y);
}
__SYCL_EXPORT s::cl_int s_min(s::cl_int x, s::cl_int y) __NOEXC {
  return std::min(x, y);
}
__SYCL_EXPORT s::cl_long s_min(s::cl_long x, s::cl_long y) __NOEXC {
  return std::min(x, y);
}
MAKE_1V_2V(s_min, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_min, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_min, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_min, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S(s_min, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S(s_min, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S(s_min, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S(s_min, s::cl_long, s::cl_long, s::cl_long)

// u_min
__SYCL_EXPORT s::cl_uchar u_min(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return std::min(x, y);
}
__SYCL_EXPORT s::cl_ushort u_min(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return std::min(x, y);
}
__SYCL_EXPORT s::cl_uint u_min(s::cl_uint x, s::cl_uint y) __NOEXC {
  return std::min(x, y);
}
__SYCL_EXPORT s::cl_ulong u_min(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return std::min(x, y);
}
MAKE_1V_2V(u_min, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_min, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_min, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_min, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S(u_min, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S(u_min, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S(u_min, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S(u_min, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// rotate
__SYCL_EXPORT s::cl_uchar rotate(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __rotate(x, y);
}
__SYCL_EXPORT s::cl_ushort rotate(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __rotate(x, y);
}
__SYCL_EXPORT s::cl_uint rotate(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __rotate(x, y);
}
__SYCL_EXPORT s::cl_ulong rotate(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __rotate(x, y);
}
__SYCL_EXPORT s::cl_char rotate(s::cl_char x, s::cl_char y) __NOEXC {
  return __rotate(x, y);
}
__SYCL_EXPORT s::cl_short rotate(s::cl_short x, s::cl_short y) __NOEXC {
  return __rotate(x, y);
}
__SYCL_EXPORT s::cl_int rotate(s::cl_int x, s::cl_int y) __NOEXC {
  return __rotate(x, y);
}
__SYCL_EXPORT s::cl_long rotate(s::cl_long x, s::cl_long y) __NOEXC {
  return __rotate(x, y);
}
MAKE_1V_2V(rotate, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(rotate, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(rotate, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(rotate, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(rotate, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(rotate, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(rotate, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(rotate, s::cl_long, s::cl_long, s::cl_long)

// u_sub_sat
__SYCL_EXPORT s::cl_uchar u_sub_sat(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __u_sub_sat(x, y);
}
__SYCL_EXPORT s::cl_ushort u_sub_sat(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __u_sub_sat(x, y);
}
__SYCL_EXPORT s::cl_uint u_sub_sat(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __u_sub_sat(x, y);
}
__SYCL_EXPORT s::cl_ulong u_sub_sat(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __u_sub_sat(x, y);
}
MAKE_1V_2V(u_sub_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_sub_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_sub_sat, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_sub_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_sub_sat
__SYCL_EXPORT s::cl_char s_sub_sat(s::cl_char x, s::cl_char y) __NOEXC {
  return __s_sub_sat(x, y);
}
__SYCL_EXPORT s::cl_short s_sub_sat(s::cl_short x, s::cl_short y) __NOEXC {
  return __s_sub_sat(x, y);
}
__SYCL_EXPORT s::cl_int s_sub_sat(s::cl_int x, s::cl_int y) __NOEXC {
  return __s_sub_sat(x, y);
}
__SYCL_EXPORT s::cl_long s_sub_sat(s::cl_long x, s::cl_long y) __NOEXC {
  return __s_sub_sat(x, y);
}
MAKE_1V_2V(s_sub_sat, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_sub_sat, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_sub_sat, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_sub_sat, s::cl_long, s::cl_long, s::cl_long)

// u_upsample
__SYCL_EXPORT s::cl_ushort u_upsample(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __upsample(x, y);
}
__SYCL_EXPORT s::cl_uint u_upsample(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __upsample(x, y);
}
__SYCL_EXPORT s::cl_ulong u_upsample(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __upsample(x, y);
}
MAKE_1V_2V(u_upsample, s::cl_ushort, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_upsample, s::cl_uint, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_upsample, s::cl_ulong, s::cl_uint, s::cl_uint)

__SYCL_EXPORT s::cl_short s_upsample(s::cl_char x, s::cl_uchar y) __NOEXC {
  return __upsample(x, y);
}
__SYCL_EXPORT s::cl_int s_upsample(s::cl_short x, s::cl_ushort y) __NOEXC {
  return __upsample(x, y);
}
__SYCL_EXPORT s::cl_long s_upsample(s::cl_int x, s::cl_uint y) __NOEXC {
  return __upsample(x, y);
}
MAKE_1V_2V(s_upsample, s::cl_short, s::cl_char, s::cl_uchar)
MAKE_1V_2V(s_upsample, s::cl_int, s::cl_short, s::cl_ushort)
MAKE_1V_2V(s_upsample, s::cl_long, s::cl_int, s::cl_uint)

// popcount
__SYCL_EXPORT s::cl_uchar popcount(s::cl_uchar x) __NOEXC {
  return __popcount(x);
}
__SYCL_EXPORT s::cl_ushort popcount(s::cl_ushort x) __NOEXC {
  return __popcount(x);
}
__SYCL_EXPORT s::cl_uint popcount(s::cl_uint x) __NOEXC {
  return __popcount(x);
}
__SYCL_EXPORT s::cl_ulong popcount(s::cl_ulong x) __NOEXC {
  return __popcount(x);
}
MAKE_1V(popcount, s::cl_uchar, s::cl_uchar)
MAKE_1V(popcount, s::cl_ushort, s::cl_ushort)
MAKE_1V(popcount, s::cl_uint, s::cl_uint)
MAKE_1V(popcount, s::cl_ulong, s::cl_ulong)

__SYCL_EXPORT s::cl_char popcount(s::cl_char x) __NOEXC {
  return __popcount(x);
}
__SYCL_EXPORT s::cl_short popcount(s::cl_short x) __NOEXC {
  return __popcount(x);
}
__SYCL_EXPORT s::cl_int popcount(s::cl_int x) __NOEXC { return __popcount(x); }
__SYCL_EXPORT s::cl_long popcount(s::cl_long x) __NOEXC {
  return __popcount(x);
}
MAKE_1V(popcount, s::cl_char, s::cl_char)
MAKE_1V(popcount, s::cl_short, s::cl_short)
MAKE_1V(popcount, s::cl_int, s::cl_int)
MAKE_1V(popcount, s::cl_long, s::cl_long)

// u_mad24
__SYCL_EXPORT s::cl_uint u_mad24(s::cl_uint x, s::cl_uint y,
                                 s::cl_uint z) __NOEXC {
  return __mad24(x, y, z);
}
MAKE_1V_2V_3V(u_mad24, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)

// s_mad24
__SYCL_EXPORT s::cl_int s_mad24(s::cl_int x, s::cl_int y, s::cl_int z) __NOEXC {
  return __mad24(x, y, z);
}
MAKE_1V_2V_3V(s_mad24, s::cl_int, s::cl_int, s::cl_int, s::cl_int)

// u_mul24
__SYCL_EXPORT s::cl_uint u_mul24(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __mul24(x, y);
}
MAKE_1V_2V(u_mul24, s::cl_uint, s::cl_uint, s::cl_uint)

// s_mul24
__SYCL_EXPORT s::cl_int s_mul24(s::cl_int x, s::cl_int y) __NOEXC {
  return __mul24(x, y);
}
MAKE_1V_2V(s_mul24, s::cl_int, s::cl_int, s::cl_int)

} // namespace __host_std
} // __SYCL_INLINE_NAMESPACE(cl)
