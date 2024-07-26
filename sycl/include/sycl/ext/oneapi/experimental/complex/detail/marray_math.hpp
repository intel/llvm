//===- marray_math.hpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"

#include <sycl/marray.hpp>

namespace sycl {
inline namespace _V1 {

namespace ext {
namespace oneapi {
namespace experimental {

#ifdef ONE_ARG_MARRAY_TYPE
#error "Multiple definition of ONE_ARG_MARRAY_TYPE"
#endif
#ifdef TWO_ARGS_MARRAY_TYPE
#error "Multiple definition of TWO_ARGS_MARRAY_TYPE"
#endif

#ifdef TWO_ARGS_POLAR_S1_MARRAY_TYPE
#error "Multiple definition of TWO_ARGS_POLAR_S1_MARRAY_TYPE"
#endif
#ifdef TWO_ARGS_POLAR_S2_MARRAY_TYPE
#error "Multiple definition of TWO_ARGS_POLAR_S2_MARRAY_TYPE"
#endif

#ifdef TWO_ARGS_POW_S1_MARRAY_TYPE
#error "Multiple definition of TWO_ARGS_POW_S1_MARRAY_TYPE"
#endif
#ifdef TWO_ARGS_POW_S2_MARRAY_TYPE
#error "Multiple definition of TWO_ARGS_POW_S2_MARRAY_TYPE"
#endif

#ifdef MARRAY_CPLX_MATH_OP
#error "Multiple definition of MARRAY_CPLX_MATH_OP"
#endif

// clang-format off
#define ONE_ARG_MARRAY_TYPE(TYPE) const sycl::marray<TYPE, NumElements> &x
#define TWO_ARGS_MARRAY_TYPE(TYPE1, TYPE2) const sycl::marray<TYPE1, NumElements> &x, const sycl::marray<TYPE2, NumElements> &y

#define TWO_ARGS_POLAR_S1_MARRAY_TYPE(TYPE1, TYPE2) const sycl::marray<TYPE1, NumElements> &x, const TYPE2 &y = 0
#define TWO_ARGS_POLAR_S2_MARRAY_TYPE(TYPE1, TYPE2) const TYPE1 &x, const sycl::marray<TYPE2, NumElements> &y

#define TWO_ARGS_POW_S1_MARRAY_TYPE(TYPE1, TYPE2) const sycl::marray<TYPE1, NumElements> &x, const TYPE2 &y
#define TWO_ARGS_POW_S2_MARRAY_TYPE(TYPE1, TYPE2) const TYPE1 &x, const sycl::marray<TYPE2, NumElements> &y

#define MARRAY_CPLX_MATH_OP(NUM_ARGS, RTN_TYPE, NAME, F, ...)                         \
template<typename T, std::size_t NumElements>                                         \
_SYCL_EXT_CPLX_INLINE_VISIBILITY                                                      \
typename std::enable_if_t<is_genfloat<T>::value, sycl::marray<RTN_TYPE, NumElements>> \
NAME(NUM_ARGS##_MARRAY_TYPE(__VA_ARGS__)) {                                           \
  sycl::marray<RTN_TYPE, NumElements> rtn;                                            \
  for (std::size_t i = 0; i < NumElements; ++i) {                                     \
    rtn[i] = F;                                                                       \
  }                                                                                   \
  return rtn;                                                                         \
}

// MARRAY_CPLX_MATH_OP(NUMBER_OF_ARGUMENTS, RETURN_TYPE, FUNCTION_NAME, FUNCTION_LOGIC,    ARGUMENTS ...
MARRAY_CPLX_MATH_OP(   ONE_ARG,             T,           abs,           abs(x[i]),         complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             T,           arg,           arg(x[i]),         complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             T,           arg,           arg(x[i]),         T);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             T,           norm,          norm(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             T,           norm,          norm(x[i]),        T);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  conj,          conj(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  conj,          conj(x[i]),        T);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  proj,          proj(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  proj,          proj(x[i]),        T);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  log,           log(x[i]),         complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  log10,         log10(x[i]),       complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  sqrt,          sqrt(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  exp,           exp(x[i]),         complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  asinh,         asinh(x[i]),       complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  acosh,         acosh(x[i]),       complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  atanh,         atanh(x[i]),       complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  sinh,          sinh(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  cosh,          cosh(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  tanh,          tanh(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  asin,          asin(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  acos,          acos(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  atan,          atan(x[i]),        complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  sin,           sin(x[i]),         complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  cos,           cos(x[i]),         complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             complex<T>,  tan,           tan(x[i]),         complex<T>);

MARRAY_CPLX_MATH_OP(   ONE_ARG,             T,           real,          x[i].real(),       complex<T>);
MARRAY_CPLX_MATH_OP(   ONE_ARG,             T,           imag,          x[i].imag(),       complex<T>);

MARRAY_CPLX_MATH_OP(   TWO_ARGS,            complex<T>,  polar,         polar(x[i], y[i]), T,          T);
MARRAY_CPLX_MATH_OP(   TWO_ARGS_POLAR_S1,   complex<T>,  polar,         polar(x[i], y),    T,          T);
MARRAY_CPLX_MATH_OP(   TWO_ARGS_POLAR_S2,   complex<T>,  polar,         polar(x, y[i]),    T,          T);

MARRAY_CPLX_MATH_OP(   TWO_ARGS,            complex<T>,  pow,           pow(x[i], y[i]),   complex<T>, T);
MARRAY_CPLX_MATH_OP(   TWO_ARGS,            complex<T>,  pow,           pow(x[i], y[i]),   complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP(   TWO_ARGS,            complex<T>,  pow,           pow(x[i], y[i]),   T,          complex<T>);

MARRAY_CPLX_MATH_OP(   TWO_ARGS_POW_S1,     complex<T>,  pow,           pow(x[i], y),      complex<T>, T);
MARRAY_CPLX_MATH_OP(   TWO_ARGS_POW_S1,     complex<T>,  pow,           pow(x[i], y),      complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP(   TWO_ARGS_POW_S1,     complex<T>,  pow,           pow(x[i], y),      T,          complex<T>);

MARRAY_CPLX_MATH_OP(   TWO_ARGS_POW_S2,     complex<T>,  pow,           pow(x, y[i]),      complex<T>, T);
MARRAY_CPLX_MATH_OP(   TWO_ARGS_POW_S2,     complex<T>,  pow,           pow(x, y[i]),      complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP(   TWO_ARGS_POW_S2,     complex<T>,  pow,           pow(x, y[i]),      T,          complex<T>);
// clang-format on

#undef ONE_ARG_MARRAY_TYPE
#undef TWO_ARGS_MARRAY_TYPE
#undef TWO_ARGS_POLAR_S1_MARRAY_TYPE
#undef TWO_ARGS_POLAR_S2_MARRAY_TYPE
#undef TWO_ARGS_POW_S1_MARRAY_TYPE
#undef TWO_ARGS_POW_S2_MARRAY_TYPE
#undef MARRAY_CPLX_MATH_OP

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace _V1
} // namespace sycl
