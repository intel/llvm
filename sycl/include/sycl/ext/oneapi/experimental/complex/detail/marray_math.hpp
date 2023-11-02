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

#ifdef MARRAY_CPLX_MATH_OP_ONE_PARAM
#error "Multiple definition of MARRAY_CPLX_MATH_OP_ONE_PARAM"
#endif

#define MARRAY_CPLX_MATH_OP_ONE_PARAM(math_func, rtn_type, arg_type)           \
  template <typename T, std::size_t NumElements>                               \
  _SYCL_EXT_CPLX_INLINE_VISIBILITY                                             \
      typename std::enable_if_t<is_genfloat<T>::value ||                       \
                                    is_gencomplex<T>::value,                   \
                                sycl::marray<rtn_type, NumElements>>           \
      math_func(const sycl::marray<arg_type, NumElements> &x) {                \
    sycl::marray<rtn_type, NumElements> rtn;                                   \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = math_func(x[i]);                                                \
    }                                                                          \
    return rtn;                                                                \
  }

MARRAY_CPLX_MATH_OP_ONE_PARAM(abs, T, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(acos, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(asin, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(atan, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(acosh, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(asinh, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(atanh, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(arg, T, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(conj, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(cos, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(cosh, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(exp, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(log, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(log10, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(norm, T, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(proj, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(proj, complex<T>, T);
MARRAY_CPLX_MATH_OP_ONE_PARAM(sin, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(sinh, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(sqrt, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(tan, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_ONE_PARAM(tanh, complex<T>, complex<T>);

#undef MARRAY_CPLX_MATH_OP_ONE_PARAM

#ifdef MARRAY_CPLX_MATH_OP_TWO_PARAM
#error "Multiple definition of MARRAY_CPLX_MATH_OP_TWO_PARAM"
#endif

#define MARRAY_CPLX_MATH_OP_TWO_PARAM(math_func, rtn_type, arg_type1,          \
                                      arg_type2)                               \
  template <typename T, std::size_t NumElements>                               \
  _SYCL_EXT_CPLX_INLINE_VISIBILITY                                             \
      typename std::enable_if_t<is_genfloat<T>::value ||                       \
                                    is_gencomplex<T>::value,                   \
                                sycl::marray<rtn_type, NumElements>>           \
      math_func(const sycl::marray<arg_type1, NumElements> &x,                 \
                const sycl::marray<arg_type2, NumElements> &y) {               \
    sycl::marray<rtn_type, NumElements> rtn;                                   \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = math_func(x[i], y[i]);                                          \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  template <typename T, std::size_t NumElements>                               \
  _SYCL_EXT_CPLX_INLINE_VISIBILITY                                             \
      typename std::enable_if_t<is_genfloat<T>::value ||                       \
                                    is_gencomplex<T>::value,                   \
                                sycl::marray<rtn_type, NumElements>>           \
      math_func(const sycl::marray<arg_type1, NumElements> &x,                 \
                const arg_type2 &y) {                                          \
    sycl::marray<rtn_type, NumElements> rtn;                                   \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = math_func(x[i], y);                                             \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  template <typename T, std::size_t NumElements>                               \
  _SYCL_EXT_CPLX_INLINE_VISIBILITY                                             \
      typename std::enable_if_t<is_genfloat<T>::value ||                       \
                                    is_gencomplex<T>::value,                   \
                                sycl::marray<rtn_type, NumElements>>           \
      math_func(const arg_type1 &x,                                            \
                const sycl::marray<arg_type2, NumElements> &y) {               \
    sycl::marray<rtn_type, NumElements> rtn;                                   \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = math_func(x, y[i]);                                             \
    }                                                                          \
    return rtn;                                                                \
  }

MARRAY_CPLX_MATH_OP_TWO_PARAM(pow, complex<T>, complex<T>, T);
MARRAY_CPLX_MATH_OP_TWO_PARAM(pow, complex<T>, complex<T>, complex<T>);
MARRAY_CPLX_MATH_OP_TWO_PARAM(pow, complex<T>, T, complex<T>);

#undef MARRAY_CPLX_MATH_OP_TWO_PARAM

// Special definition as polar requires default argument

template <typename T, std::size_t NumElements>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<T>::value,
                              sycl::marray<complex<T>, NumElements>>
    polar(const sycl::marray<T, NumElements> &rho,
          const sycl::marray<T, NumElements> &theta) {
  sycl::marray<complex<T>, NumElements> rtn;
  for (std::size_t i = 0; i < NumElements; ++i) {
    rtn[i] = polar(rho[i], theta[i]);
  }
  return rtn;
}

template <typename T, std::size_t NumElements>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<T>::value,
                              sycl::marray<complex<T>, NumElements>>
    polar(const sycl::marray<T, NumElements> &rho, const T &theta = 0) {
  sycl::marray<complex<T>, NumElements> rtn;
  for (std::size_t i = 0; i < NumElements; ++i) {
    rtn[i] = polar(rho[i], theta);
  }
  return rtn;
}

template <typename T, std::size_t NumElements>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<T>::value,
                              sycl::marray<complex<T>, NumElements>>
    polar(const T &rho, const sycl::marray<T, NumElements> &theta) {
  sycl::marray<complex<T>, NumElements> rtn;
  for (std::size_t i = 0; i < NumElements; ++i) {
    rtn[i] = polar(rho, theta[i]);
  }
  return rtn;
}

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace _V1
} // namespace sycl
