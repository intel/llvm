//==------------------- math_functions.cpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Define _USE_MATH_DEFINES to enforce math defines of macros like M_PI in
// <cmath>. _USE_MATH_DEFINES is defined here before includes of SYCL header
// files to avoid include of <cmath> via those SYCL headers with unset
// _USE_MATH_DEFINES.
#define _USE_MATH_DEFINES

#include <cmath>

#include <sycl/builtins_preview.hpp>

#include "host_helper_macros.hpp"

namespace sycl {
inline namespace _V1 {
#define BUILTIN_GENF_CUSTOM(NUM_ARGS, NAME, IMPL)                              \
  HOST_IMPL(NAME, IMPL)                                                        \
  EXPORT_SCALAR_AND_VEC_1_16(NUM_ARGS, NAME, FP_TYPES)

// NOTE: "-> decltype(x)" here and below is need for the half version, what
// implementation do is invoking implicit conversion to float and compute the
// result in float precision. Need to convert back by specifying return type.
#define BUILTIN_GENF(NUM_ARGS, NAME)                                           \
  BUILTIN_GENF_CUSTOM(NUM_ARGS, NAME, [](NUM_ARGS##_AUTO_ARG) -> decltype(x) { \
    return std::NAME(NUM_ARGS##_ARG);                                          \
  })

BUILTIN_GENF(ONE_ARG, acos)
BUILTIN_GENF(ONE_ARG, acosh)
BUILTIN_GENF_CUSTOM(ONE_ARG, acospi,
                    [](auto x) -> decltype(x) { return std::acos(x) / M_PI; })
BUILTIN_GENF(ONE_ARG, asin)
BUILTIN_GENF(ONE_ARG, asinh)
BUILTIN_GENF_CUSTOM(ONE_ARG, asinpi,
                    [](auto x) -> decltype(x) { return std::asin(x) / M_PI; })
BUILTIN_GENF(ONE_ARG, atan)
BUILTIN_GENF(ONE_ARG, atanh)
BUILTIN_GENF_CUSTOM(ONE_ARG, atanpi,
                    [](auto x) -> decltype(x) { return std::atan(x) / M_PI; })
BUILTIN_GENF(TWO_ARGS, atan2)
BUILTIN_GENF_CUSTOM(TWO_ARGS, atan2pi, [](auto x, auto y) -> decltype(x) {
  return std::atan2(x, y) / M_PI;
})
BUILTIN_GENF(ONE_ARG, cbrt)
BUILTIN_GENF(ONE_ARG, ceil)
BUILTIN_GENF(TWO_ARGS, copysign)
BUILTIN_GENF(ONE_ARG, cos)
BUILTIN_GENF(ONE_ARG, cosh)
BUILTIN_GENF_CUSTOM(ONE_ARG, cospi, [](auto x) -> decltype(x) {
  return std::sin(M_PI * (0.5 - x));
})
BUILTIN_GENF(ONE_ARG, erf)
BUILTIN_GENF(ONE_ARG, erfc)
BUILTIN_GENF(ONE_ARG, exp)
BUILTIN_GENF(ONE_ARG, exp2)
BUILTIN_GENF_CUSTOM(ONE_ARG, exp10,
                    [](auto x) -> decltype(x) { return std::pow(10, x); })
BUILTIN_GENF(ONE_ARG, expm1)
BUILTIN_GENF(ONE_ARG, fabs)
BUILTIN_GENF(TWO_ARGS, fdim)
BUILTIN_GENF(ONE_ARG, floor)
BUILTIN_GENF(THREE_ARGS, fma)
BUILTIN_GENF(TWO_ARGS, fmax)
BUILTIN_GENF(TWO_ARGS, fmin)
BUILTIN_GENF(TWO_ARGS, fmod)
BUILTIN_GENF(TWO_ARGS, hypot)
BUILTIN_GENF(ONE_ARG, lgamma)
BUILTIN_GENF(ONE_ARG, log)
BUILTIN_GENF(ONE_ARG, log2)
BUILTIN_GENF(ONE_ARG, log10)
BUILTIN_GENF(ONE_ARG, log1p)
BUILTIN_GENF(ONE_ARG, logb)
BUILTIN_GENF_CUSTOM(THREE_ARGS, mad, [](auto x, auto y, auto z) -> decltype(x) {
  return (x * y) + z;
})
BUILTIN_GENF_CUSTOM(TWO_ARGS, maxmag, [](auto x, auto y) -> decltype(x) {
  if (std::fabs(x) > std::fabs(y))
    return x;
  if (std::fabs(y) > std::fabs(x))
    return y;
  return std::fmax(x, y);
})
BUILTIN_GENF_CUSTOM(TWO_ARGS, minmag, [](auto x, auto y) -> decltype(x) {
  if (std::fabs(x) < std::fabs(y))
    return x;
  if (std::fabs(y) < std::fabs(x))
    return y;
  return std::fmin(x, y);
})
BUILTIN_GENF(TWO_ARGS, pow)
BUILTIN_GENF_CUSTOM(TWO_ARGS, powr, [](auto x, auto y) -> decltype(x) {
  using T = decltype(x);
  return (x >= T(0)) ? T(std::pow(x, y)) : x;
})
BUILTIN_GENF(TWO_ARGS, remainder)
BUILTIN_GENF(ONE_ARG, rint)
BUILTIN_GENF(ONE_ARG, round)
BUILTIN_GENF_CUSTOM(ONE_ARG, rsqrt, [](auto x) -> decltype(x) {
  return decltype(x){1.0} / std::sqrt(x);
})
BUILTIN_GENF(ONE_ARG, sin)
BUILTIN_GENF(ONE_ARG, sinh)
BUILTIN_GENF_CUSTOM(ONE_ARG, sinpi,
                    [](auto x) -> decltype(x) { return std::sin(M_PI * x); })
BUILTIN_GENF(ONE_ARG, sqrt)
BUILTIN_GENF(ONE_ARG, tan)
BUILTIN_GENF(ONE_ARG, tanh)
BUILTIN_GENF_CUSTOM(
    ONE_ARG, tanpi,
    [](auto x) -> decltype(x) { // For uniformity, place in range [0.0, 1.0).
      double y = x - std::floor(x);
      // Flip for better accuracy.
      return 1.0 / std::tan((0.5 - y) * M_PI);
    })
BUILTIN_GENF(ONE_ARG, tgamma)
BUILTIN_GENF(ONE_ARG, trunc)
BUILTIN_GENF_CUSTOM(TWO_ARGS, nextafter, [](auto x, auto y) {
  if constexpr (!std::is_same_v<decltype(x), half>) {
    return std::nextafter(x, y);
  } else {
    // Copied from sycl_host_nextafter, not sure if it's valid when operating on
    // sycl::half. That said, should be covered by
    // sycl/test/regression/host_half_nextafter.cpp

    if (std::isnan(static_cast<float>(x)))
      return x;
    if (std::isnan(static_cast<float>(y)) || x == y)
      return y;

    uint16_t x_bits = sycl::bit_cast<uint16_t>(x);
    uint16_t x_sign = x_bits & 0x8000;
    int16_t movement = (x > y ? -1 : 1) * (x_sign ? -1 : 1);
    if (x_bits == x_sign && movement == -1) {
      // Special case where we underflow in the decrement, in which case we turn
      // it around and flip the sign. The overflow case does not need special
      // handling.
      movement = 1;
      x_bits ^= 0x8000;
    }
    x_bits += movement;
    return sycl::bit_cast<half>(x_bits);
  }
})

namespace detail {
__SYCL_EXPORT float frexp_impl(float x, int *p) { return std::frexp(x, p); }
__SYCL_EXPORT double frexp_impl(double x, int *p) { return std::frexp(x, p); }
__SYCL_EXPORT half frexp_impl(half x, int *p) { return std::frexp(x, p); }
} // namespace detail

namespace detail {
template <typename T> static inline T __lgamma_r_impl(T x, int *signp) {
  T g = std::tgamma(x);
  *signp = std::signbit(sycl::detail::cast_if_host_half(g)) ? -1 : 1;
  return std::log(std::abs(g));
}

__SYCL_EXPORT float lgamma_r_impl(float x, int *p) {
  return __lgamma_r_impl(x, p);
}
__SYCL_EXPORT double lgamma_r_impl(double x, int *p) {
  return __lgamma_r_impl(x, p);
}
__SYCL_EXPORT half lgamma_r_impl(half x, int *p) {
  return __lgamma_r_impl(x, p);
}
} // namespace detail

HOST_IMPL(ilogb, std::ilogb)
EXPORT_SCALAR_AND_VEC_1_16(ONE_ARG, ilogb, FP_TYPES)

namespace detail {
__SYCL_EXPORT float modf_impl(float x, float *p) { return std::modf(x, p); }
__SYCL_EXPORT double modf_impl(double x, double *p) { return std::modf(x, p); }
__SYCL_EXPORT half modf_impl(half x, half *p) {
  float val;
  auto ret = std::modf(x, &val);
  *p = val;
  return ret;
}
} // namespace detail

namespace detail {
template <typename T> static inline T __sincos(T x, T *cosval) {
  (*cosval) = std::cos(x);
  return std::sin(x);
}

__SYCL_EXPORT float sincos_impl(float x, float *p) { return __sincos(x, p); }
__SYCL_EXPORT double sincos_impl(double x, double *p) { return __sincos(x, p); }
__SYCL_EXPORT half sincos_impl(half x, half *p) { return __sincos(x, p); }
} // namespace detail

#define EXPORT_VEC_LAST_INT(NAME, TYPE, VL)                                    \
  vec<TYPE, VL> __SYCL_EXPORT __##NAME##_impl(vec<TYPE, VL> x,                 \
                                              vec<int, VL> y) {                \
    return NAME##_host_impl(x, y);                                             \
  }
#define EXPORT_VEC_LAST_INT_1_16(NAME, TYPE)                                   \
  FOR_VEC_1_16(EXPORT_VEC_LAST_INT, NAME, TYPE)

#define BUILTIN_MATH_LAST_INT(NAME, IMPL)                                      \
  __SYCL_EXPORT float __##NAME##_impl(float x, int y) { return IMPL(x, y); }   \
  __SYCL_EXPORT double __##NAME##_impl(double x, int y) { return IMPL(x, y); } \
  __SYCL_EXPORT half __##NAME##_impl(half x, int y) { return IMPL(x, y); }     \
  HOST_IMPL(NAME, NAME /* delegate to scalar */)                               \
  FOR_EACH1(EXPORT_VEC_LAST_INT_1_16, NAME, FP_TYPES)

BUILTIN_MATH_LAST_INT(pown, std::pow)
BUILTIN_MATH_LAST_INT(rootn, [](auto x, auto y) -> decltype(x) {
  return std::pow(x, decltype(x){1} / y);
})
BUILTIN_MATH_LAST_INT(ldexp, std::ldexp)

namespace {
template <typename T> auto __remquo_impl(T x, T y, int *z) {
  T rem = std::remainder(x, y);
  *z = static_cast<int>(std::round((x - rem) / y));
  return rem;
}
} // namespace
namespace detail {
__SYCL_EXPORT float remquo_impl(float x, float y, int *z) {
  return __remquo_impl(x, y, z);
}
__SYCL_EXPORT double remquo_impl(double x, double y, int *z) {
  return __remquo_impl(x, y, z);
}
__SYCL_EXPORT half remquo_impl(half x, half y, int *z) {
  return __remquo_impl(x, y, z);
}
} // namespace detail
} // namespace _V1
} // namespace sycl
