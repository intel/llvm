//==----------- builtins_common.cpp - SYCL built-in common functions -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines the host versions of functions defined
// in SYCL SPEC section - 4.13.5 Common functions.

// Define _USE_MATH_DEFINES to enforce math defines of macros like M_PI in
// <cmath>. _USE_MATH_DEFINES is defined here before includes of SYCL header
// files to avoid include of <cmath> via those SYCL headers with unset
// _USE_MATH_DEFINES.
#define _USE_MATH_DEFINES

#include "builtins_helper.hpp"

#include <cmath>

namespace s = sycl;
namespace d = s::detail;

namespace __host_std {
namespace {

template <typename T> inline T __fclamp(T x, T minval, T maxval) {
  return std::fmin(std::fmax(x, minval), maxval);
}

template <typename T> inline T __degrees(T radians) {
  return (180 / M_PI) * radians;
}

template <typename T> inline T __mix(T x, T y, T a) { return x + (y - x) * a; }

template <typename T> inline T __radians(T degrees) {
  return (M_PI / 180) * degrees;
}

template <typename T> inline T __step(T edge, T x) {
  return (x < edge) ? 0.0 : 1.0;
}

template <typename T> inline T __smoothstep(T edge0, T edge1, T x) {
  T t;
  T v = (x - edge0) / (edge1 - edge0);
  t = __fclamp(v, T(0), T(1));
  return t * t * (3 - 2 * t);
}

template <typename T> inline T __sign(T x) {
  if (std::isnan(d::cast_if_host_half(x)))
    return T(0.0);
  if (x > 0)
    return T(1.0);
  if (x < 0)
    return T(-1.0);
  /* x is +0.0 or -0.0 */
  return x;
}

} // namespace

// --------------- 4.13.5 Common functions. Host implementations ---------------
// fclamp
__SYCL_EXPORT s::cl_float sycl_host_fclamp(s::cl_float x, s::cl_float minval,
                                           s::cl_float maxval) __NOEXC {
  return __fclamp(x, minval, maxval);
}
__SYCL_EXPORT s::cl_double sycl_host_fclamp(s::cl_double x, s::cl_double minval,
                                            s::cl_double maxval) __NOEXC {
  return __fclamp(x, minval, maxval);
}
__SYCL_EXPORT s::cl_half sycl_host_fclamp(s::cl_half x, s::cl_half minval,
                                          s::cl_half maxval) __NOEXC {
  return __fclamp(x, minval, maxval);
}
MAKE_1V_2V_3V(sycl_host_fclamp, s::cl_float, s::cl_float, s::cl_float,
              s::cl_float)
MAKE_1V_2V_3V(sycl_host_fclamp, s::cl_double, s::cl_double, s::cl_double,
              s::cl_double)
MAKE_1V_2V_3V(sycl_host_fclamp, s::cl_half, s::cl_half, s::cl_half, s::cl_half)

// degrees
__SYCL_EXPORT s::cl_float sycl_host_degrees(s::cl_float radians) __NOEXC {
  return __degrees(radians);
}
__SYCL_EXPORT s::cl_double sycl_host_degrees(s::cl_double radians) __NOEXC {
  return __degrees(radians);
}
__SYCL_EXPORT s::cl_half sycl_host_degrees(s::cl_half radians) __NOEXC {
  return __degrees(radians);
}
MAKE_1V(sycl_host_degrees, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_degrees, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_degrees, s::cl_half, s::cl_half)

// fmin_common
__SYCL_EXPORT s::cl_float sycl_host_fmin_common(s::cl_float x,
                                                s::cl_float y) __NOEXC {
  return std::fmin(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_fmin_common(s::cl_double x,
                                                 s::cl_double y) __NOEXC {
  return std::fmin(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_fmin_common(s::cl_half x,
                                               s::cl_half y) __NOEXC {
  return std::fmin(x, y);
}
MAKE_1V_2V(sycl_host_fmin_common, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_fmin_common, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_fmin_common, s::cl_half, s::cl_half, s::cl_half)

// fmax_common
__SYCL_EXPORT s::cl_float sycl_host_fmax_common(s::cl_float x,
                                                s::cl_float y) __NOEXC {
  return std::fmax(x, y);
}
__SYCL_EXPORT s::cl_double sycl_host_fmax_common(s::cl_double x,
                                                 s::cl_double y) __NOEXC {
  return std::fmax(x, y);
}
__SYCL_EXPORT s::cl_half sycl_host_fmax_common(s::cl_half x,
                                               s::cl_half y) __NOEXC {
  return std::fmax(x, y);
}
MAKE_1V_2V(sycl_host_fmax_common, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_fmax_common, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_fmax_common, s::cl_half, s::cl_half, s::cl_half)

// mix
__SYCL_EXPORT s::cl_float sycl_host_mix(s::cl_float x, s::cl_float y,
                                        s::cl_float a) __NOEXC {
  return __mix(x, y, a);
}
__SYCL_EXPORT s::cl_double sycl_host_mix(s::cl_double x, s::cl_double y,
                                         s::cl_double a) __NOEXC {
  return __mix(x, y, a);
}
__SYCL_EXPORT s::cl_half sycl_host_mix(s::cl_half x, s::cl_half y,
                                       s::cl_half a) __NOEXC {
  return __mix(x, y, a);
}
MAKE_1V_2V_3V(sycl_host_mix, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(sycl_host_mix, s::cl_double, s::cl_double, s::cl_double,
              s::cl_double)
MAKE_1V_2V_3V(sycl_host_mix, s::cl_half, s::cl_half, s::cl_half, s::cl_half)

// radians
__SYCL_EXPORT s::cl_float sycl_host_radians(s::cl_float degrees) __NOEXC {
  return __radians(degrees);
}
__SYCL_EXPORT s::cl_double sycl_host_radians(s::cl_double degrees) __NOEXC {
  return __radians(degrees);
}
__SYCL_EXPORT s::cl_half sycl_host_radians(s::cl_half degrees) __NOEXC {
  return __radians(degrees);
}
MAKE_1V(sycl_host_radians, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_radians, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_radians, s::cl_half, s::cl_half)

// step
__SYCL_EXPORT s::cl_float sycl_host_step(s::cl_float edge,
                                         s::cl_float x) __NOEXC {
  return __step(edge, x);
}
__SYCL_EXPORT s::cl_double sycl_host_step(s::cl_double edge,
                                          s::cl_double x) __NOEXC {
  return __step(edge, x);
}
__SYCL_EXPORT s::cl_half sycl_host_step(s::cl_half edge, s::cl_half x) __NOEXC {
  return __step(edge, x);
}
MAKE_1V_2V(sycl_host_step, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(sycl_host_step, s::cl_double, s::cl_double, s::cl_double)
MAKE_1V_2V(sycl_host_step, s::cl_half, s::cl_half, s::cl_half)

// smoothstep
__SYCL_EXPORT s::cl_float sycl_host_smoothstep(s::cl_float edge0,
                                               s::cl_float edge1,
                                               s::cl_float x) __NOEXC {
  return __smoothstep(edge0, edge1, x);
}
__SYCL_EXPORT s::cl_double sycl_host_smoothstep(s::cl_double edge0,
                                                s::cl_double edge1,
                                                s::cl_double x) __NOEXC {
  return __smoothstep(edge0, edge1, x);
}
__SYCL_EXPORT s::cl_half
sycl_host_smoothstep(s::cl_half edge0, s::cl_half edge1, s::cl_half x) __NOEXC {
  return __smoothstep(edge0, edge1, x);
}
MAKE_1V_2V_3V(sycl_host_smoothstep, s::cl_float, s::cl_float, s::cl_float,
              s::cl_float)
MAKE_1V_2V_3V(sycl_host_smoothstep, s::cl_double, s::cl_double, s::cl_double,
              s::cl_double)
MAKE_1V_2V_3V(sycl_host_smoothstep, s::cl_half, s::cl_half, s::cl_half,
              s::cl_half)

// sign
__SYCL_EXPORT s::cl_float sycl_host_sign(s::cl_float x) __NOEXC {
  return __sign(x);
}
__SYCL_EXPORT s::cl_double sycl_host_sign(s::cl_double x) __NOEXC {
  return __sign(x);
}
__SYCL_EXPORT s::cl_half sycl_host_sign(s::cl_half x) __NOEXC {
  return __sign(x);
}
MAKE_1V(sycl_host_sign, s::cl_float, s::cl_float)
MAKE_1V(sycl_host_sign, s::cl_double, s::cl_double)
MAKE_1V(sycl_host_sign, s::cl_half, s::cl_half)

} // namespace __host_std
