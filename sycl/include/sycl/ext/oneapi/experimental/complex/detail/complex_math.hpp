//===- complex_math.hpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"

#include <sycl/builtins.hpp>

#include <math.h>

namespace sycl {
inline namespace _V1 {

namespace ext {
namespace oneapi {
namespace experimental {

////////////////////////////////////////////////////////////////////////////////
/// TRAITS
////////////////////////////////////////////////////////////////////////////////

namespace cplx::detail {

template <bool _Val> using _BoolConstant = std::integral_constant<bool, _Val>;

template <class _Tp, class _Up>
using _IsNotSame = _BoolConstant<!__is_same(_Tp, _Up)>;

template <class _Tp> struct __numeric_type {
  static void __test(...);
  static sycl::half __test(sycl::half);
  static float __test(float);
  static double __test(char);
  static double __test(int);
  static double __test(unsigned);
  static double __test(long);
  static double __test(unsigned long);
  static double __test(long long);
  static double __test(unsigned long long);
  static double __test(double);

  typedef decltype(__test(std::declval<_Tp>())) type;
  static const bool value = _IsNotSame<type, void>::value;
};

template <> struct __numeric_type<void> {
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value && __numeric_type<_A2>::value &&
                 __numeric_type<_A3>::value>
class __promote_imp {
public:
  static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;
  typedef typename __promote_imp<_A3>::type __type3;

public:
  typedef decltype(__type1() + __type2() + __type3()) type;
  static const bool value = true;
};

template <class _A1, class _A2> class __promote_imp<_A1, _A2, void, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;

public:
  typedef decltype(__type1() + __type2()) type;
  static const bool value = true;
};

template <class _A1> class __promote_imp<_A1, void, void, true> {
public:
  typedef typename __numeric_type<_A1>::type type;
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

template <class _Tp, bool = std::is_integral<_Tp>::value,
          bool = is_genfloat<_Tp>::value>
struct __libcpp_complex_overload_traits {};

// Integral Types
template <class _Tp> struct __libcpp_complex_overload_traits<_Tp, true, false> {
  typedef double _ValueType;
  typedef complex<double> _ComplexType;
};

// Floating point types
template <class _Tp> struct __libcpp_complex_overload_traits<_Tp, false, true> {
  typedef _Tp _ValueType;
  typedef complex<_Tp> _ComplexType;
};

} // namespace cplx::detail

////////////////////////////////////////////////////////////////////////////////
/// FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

// abs

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, _Tp>
    abs(const complex<_Tp> &__c) {
  return sycl::hypot(__c.real(), __c.imag());
}

// arg

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, _Tp>
    arg(const complex<_Tp> &__c) {
  return sycl::atan2(__c.imag(), __c.real());
}

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    arg(_Tp __re) {
  typedef
      typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
          _ValueType;
  return sycl::atan2(static_cast<_ValueType>(0), static_cast<_ValueType>(__re));
}

// norm

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, _Tp>
    norm(const complex<_Tp> &__c) {
  if (sycl::isinf(__c.real()))
    return sycl::fabs(__c.real());
  if (sycl::isinf(__c.imag()))
    return sycl::fabs(__c.imag());
  return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    norm(_Tp __re) {
  typedef
      typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
          _ValueType;
  return static_cast<_ValueType>(__re) * __re;
}

// conj

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    conj(const complex<_Tp> &__c) {
  return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType
    conj(_Tp __re) {
  typedef
      typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType
          _ComplexType;
  return _ComplexType(__re);
}

// proj

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    proj(const complex<_Tp> &__c) {
  complex<_Tp> __r = __c;
  if (sycl::isinf(__c.real()) || sycl::isinf(__c.imag()))
    __r = complex<_Tp>(INFINITY, sycl::copysign(_Tp(0), __c.imag()));
  return __r;
}

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType
    proj(_Tp __re) {
  typedef
      typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ComplexType
          _ComplexType;

  if constexpr (!std::is_integral_v<_Tp>) {
    if (sycl::isinf(__re))
      __re = sycl::fabs(__re);
  }

  return _ComplexType(__re);
}

// polar

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    polar(const _Tp &__rho, const _Tp &__theta = _Tp()) {
  if (sycl::isnan(__rho) || sycl::signbit(__rho))
    return complex<_Tp>(_Tp(NAN), _Tp(NAN));
  if (sycl::isnan(__theta)) {
    if (sycl::isinf(__rho))
      return complex<_Tp>(__rho, __theta);
    return complex<_Tp>(__theta, __theta);
  }
  if (sycl::isinf(__theta)) {
    if (sycl::isinf(__rho))
      return complex<_Tp>(__rho, _Tp(NAN));
    return complex<_Tp>(_Tp(NAN), _Tp(NAN));
  }
  _Tp __x = __rho * sycl::cos(__theta);
  if (sycl::isnan(__x))
    __x = 0;
  _Tp __y = __rho * sycl::sin(__theta);
  if (sycl::isnan(__y))
    __y = 0;
  return complex<_Tp>(__x, __y);
}

// log

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    log(const complex<_Tp> &__x) {
  return complex<_Tp>(sycl::log(abs(__x)), arg(__x));
}

// log10

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    log10(const complex<_Tp> &__x) {
  return log(__x) / sycl::log(_Tp(10));
}

// sqrt

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    sqrt(const complex<_Tp> &__x) {
  if (sycl::isinf(__x.imag()))
    return complex<_Tp>(_Tp(INFINITY), __x.imag());
  if (sycl::isinf(__x.real())) {
    if (__x.real() > _Tp(0))
      return complex<_Tp>(__x.real(), sycl::isnan(__x.imag())
                                          ? __x.imag()
                                          : sycl::copysign(_Tp(0), __x.imag()));
    return complex<_Tp>(sycl::isnan(__x.imag()) ? __x.imag() : _Tp(0),
                        sycl::copysign(__x.real(), __x.imag()));
  }
  return polar(sycl::sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    exp(const complex<_Tp> &__x) {
  _Tp __i = __x.imag();
  if (__i == 0) {
    return complex<_Tp>(sycl::exp(__x.real()),
                        sycl::copysign(_Tp(0), __x.imag()));
  }
  if (sycl::isinf(__x.real())) {
    if (__x.real() < _Tp(0)) {
      if (!sycl::isfinite(__i))
        __i = _Tp(1);
    } else if (__i == 0 || !sycl::isfinite(__i)) {
      if (sycl::isinf(__i))
        __i = _Tp(NAN);
      return complex<_Tp>(__x.real(), __i);
    }
  }
  _Tp __e = sycl::exp(__x.real());
  return complex<_Tp>(__e * sycl::cos(__i), __e * sycl::sin(__i));
}

// pow

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    pow(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  return exp(__y * log(__x));
}

template <class _Tp, class _Up>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<
        is_genfloat<_Tp>::value,
        complex<typename cplx::detail::__promote<_Tp, _Up>::type>>
    pow(const complex<_Tp> &__x, const complex<_Up> &__y) {
  using result_type = complex<typename cplx::detail::__promote<_Tp, _Up>::type>;
  return pow(result_type(__x), result_type(__y));
}

template <class _Tp, class _Up>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<
        is_genfloat<_Tp>::value && is_genfloat<_Up>::value,
        complex<typename cplx::detail::__promote<_Tp, _Up>::type>>
    pow(const complex<_Tp> &__x, const _Up &__y) {
  using result_type = complex<typename cplx::detail::__promote<_Tp, _Up>::type>;
  return pow(result_type(__x), result_type(__y));
}

template <class _Tp, class _Up>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<
        is_genfloat<_Tp>::value && is_genfloat<_Up>::value,
        complex<typename cplx::detail::__promote<_Tp, _Up>::type>>
    pow(const _Tp &__x, const complex<_Up> &__y) {
  using result_type = complex<typename cplx::detail::__promote<_Tp, _Up>::type>;
  return pow(result_type(__x), result_type(__y));
}

namespace cplx::detail {
// __sqr, computes pow(x, 2)

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    __sqr(const complex<_Tp> &__x) {
  return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()),
                      _Tp(2) * __x.real() * __x.imag());
}
} // namespace cplx::detail

// asinh

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    asinh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(_Tp(+0.), _Tp(-0.)));
  if (sycl::isinf(__x.real())) {
    if (sycl::isnan(__x.imag()))
      return __x;
    if (sycl::isinf(__x.imag()))
      return complex<_Tp>(__x.real(),
                          sycl::copysign(__pi * _Tp(0.25), __x.imag()));
    return complex<_Tp>(__x.real(), sycl::copysign(_Tp(0), __x.imag()));
  }
  if (sycl::isnan(__x.real())) {
    if (sycl::isinf(__x.imag()))
      return complex<_Tp>(__x.imag(), __x.real());
    if (__x.imag() == 0)
      return __x;
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (sycl::isinf(__x.imag()))
    return complex<_Tp>(sycl::copysign(__x.imag(), __x.real()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  complex<_Tp> __z = log(__x + sqrt(cplx::detail::__sqr(__x) + _Tp(1)));
  return complex<_Tp>(sycl::copysign(__z.real(), __x.real()),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// acosh

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    acosh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(_Tp(+0.), _Tp(-0.)));
  if (sycl::isinf(__x.real())) {
    if (sycl::isnan(__x.imag()))
      return complex<_Tp>(sycl::fabs(__x.real()), __x.imag());
    if (sycl::isinf(__x.imag())) {
      if (__x.real() > 0)
        return complex<_Tp>(__x.real(),
                            sycl::copysign(__pi * _Tp(0.25), __x.imag()));
      else
        return complex<_Tp>(-__x.real(),
                            sycl::copysign(__pi * _Tp(0.75), __x.imag()));
    }
    if (__x.real() < 0)
      return complex<_Tp>(-__x.real(), sycl::copysign(__pi, __x.imag()));
    return complex<_Tp>(__x.real(), sycl::copysign(_Tp(0), __x.imag()));
  }
  if (sycl::isnan(__x.real())) {
    if (sycl::isinf(__x.imag()))
      return complex<_Tp>(sycl::fabs(__x.imag()), __x.real());
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (sycl::isinf(__x.imag()))
    return complex<_Tp>(sycl::fabs(__x.imag()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  complex<_Tp> __z = log(__x + sqrt(cplx::detail::__sqr(__x) - _Tp(1)));
  return complex<_Tp>(sycl::copysign(__z.real(), _Tp(0)),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// atanh

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    atanh(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(_Tp(+0.), _Tp(-0.)));
  if (sycl::isinf(__x.imag())) {
    return complex<_Tp>(sycl::copysign(_Tp(0), __x.real()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  }
  if (sycl::isnan(__x.imag())) {
    if (sycl::isinf(__x.real()) || __x.real() == 0)
      return complex<_Tp>(sycl::copysign(_Tp(0), __x.real()), __x.imag());
    return complex<_Tp>(__x.imag(), __x.imag());
  }
  if (sycl::isnan(__x.real())) {
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (sycl::isinf(__x.real())) {
    return complex<_Tp>(sycl::copysign(_Tp(0), __x.real()),
                        sycl::copysign(__pi / _Tp(2), __x.imag()));
  }
  if (sycl::fabs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0)) {
    return complex<_Tp>(sycl::copysign(_Tp(INFINITY), __x.real()),
                        sycl::copysign(_Tp(0), __x.imag()));
  }
  complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
  return complex<_Tp>(sycl::copysign(__z.real(), __x.real()),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// sinh

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    sinh(const complex<_Tp> &__x) {
  if (sycl::isinf(__x.real()) && !sycl::isfinite(__x.imag()))
    return complex<_Tp>(__x.real(), _Tp(NAN));
  if (__x.real() == 0 && !sycl::isfinite(__x.imag()))
    return complex<_Tp>(__x.real(), _Tp(NAN));
  if (__x.imag() == 0 && !sycl::isfinite(__x.real()))
    return __x;
  return complex<_Tp>(sycl::sinh(__x.real()) * sycl::cos(__x.imag()),
                      sycl::cosh(__x.real()) * sycl::sin(__x.imag()));
}

// cosh

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    cosh(const complex<_Tp> &__x) {
  if (sycl::isinf(__x.real()) && !sycl::isfinite(__x.imag()))
    return complex<_Tp>(sycl::fabs(__x.real()), _Tp(NAN));
  if (__x.real() == 0 && !sycl::isfinite(__x.imag()))
    return complex<_Tp>(_Tp(NAN), __x.real());
  if (__x.real() == 0 && __x.imag() == 0)
    return complex<_Tp>(_Tp(1), __x.imag());
  if (__x.imag() == 0 && !sycl::isfinite(__x.real()))
    return complex<_Tp>(sycl::fabs(__x.real()), __x.imag());
  return complex<_Tp>(sycl::cosh(__x.real()) * sycl::cos(__x.imag()),
                      sycl::sinh(__x.real()) * sycl::sin(__x.imag()));
}

// tanh

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    tanh(const complex<_Tp> &__x) {
  if (sycl::isinf(__x.real())) {
    if (!sycl::isfinite(__x.imag()))
      return complex<_Tp>(sycl::copysign(_Tp(1), __x.real()), _Tp(0));
    return complex<_Tp>(sycl::copysign(_Tp(1), __x.real()),
                        sycl::copysign(_Tp(0), sycl::sin(_Tp(2) * __x.imag())));
  }
  if (sycl::isnan(__x.real()) && __x.imag() == 0)
    return __x;
  _Tp __2r(_Tp(2) * __x.real());
  _Tp __2i(_Tp(2) * __x.imag());
  _Tp __d(sycl::cosh(__2r) + sycl::cos(__2i));
  _Tp __2rsh(sycl::sinh(__2r));
  if (sycl::isinf(__2rsh) && sycl::isinf(__d))
    return complex<_Tp>(__2rsh > _Tp(0) ? _Tp(1) : _Tp(-1),
                        __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
  return complex<_Tp>(__2rsh / __d, sycl::sin(__2i) / __d);
}

// asin

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    asin(const complex<_Tp> &__x) {
  complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    acos(const complex<_Tp> &__x) {
  const _Tp __pi(sycl::atan2(_Tp(+0.), _Tp(-0.)));
  if (sycl::isinf(__x.real())) {
    if (sycl::isnan(__x.imag()))
      return complex<_Tp>(__x.imag(), __x.real());
    if (sycl::isinf(__x.imag())) {
      if (__x.real() < _Tp(0))
        return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
      return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
    }
    if (__x.real() < _Tp(0))
      return complex<_Tp>(__pi,
                          sycl::signbit(__x.imag()) ? -__x.real() : __x.real());
    return complex<_Tp>(_Tp(0),
                        sycl::signbit(__x.imag()) ? __x.real() : -__x.real());
  }
  if (sycl::isnan(__x.real())) {
    if (sycl::isinf(__x.imag()))
      return complex<_Tp>(__x.real(), -__x.imag());
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (sycl::isinf(__x.imag()))
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  if (__x.real() == 0 && (__x.imag() == 0 || sycl::isnan(__x.imag())))
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  complex<_Tp> __z = log(__x + sqrt(cplx::detail::__sqr(__x) - _Tp(1)));
  if (sycl::signbit(__x.imag()))
    return complex<_Tp>(sycl::fabs(__z.imag()), sycl::fabs(__z.real()));
  return complex<_Tp>(sycl::fabs(__z.imag()), -sycl::fabs(__z.real()));
}

// atan

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    atan(const complex<_Tp> &__x) {
  complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    sin(const complex<_Tp> &__x) {
  complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    cos(const complex<_Tp> &__x) {
  return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename std::enable_if_t<is_genfloat<_Tp>::value, complex<_Tp>>
    tan(const complex<_Tp> &__x) {
  complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// real

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename std::enable_if_t<is_genfloat<_Tp>::value, _Tp>
    real(const complex<_Tp> &__c) {
  return __c.real();
}

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    real(_Tp __re) {
  return __re;
}

// imag

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename std::enable_if_t<is_genfloat<_Tp>::value, _Tp>
    imag(const complex<_Tp> &__c) {
  return __c.imag();
}

template <class _Tp>
__DPCPP_SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename cplx::detail::__libcpp_complex_overload_traits<_Tp>::_ValueType
    imag(_Tp) {
  return 0;
}

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace _V1
} // namespace sycl
