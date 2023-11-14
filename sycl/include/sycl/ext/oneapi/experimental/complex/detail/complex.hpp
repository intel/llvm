//===- complex.hpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"

#include <sycl/stream.hpp>

#include <complex>

namespace sycl {
inline namespace _V1 {

namespace ext {
namespace oneapi {
namespace experimental {

template <class _Tp>
class complex<_Tp, typename std::enable_if_t<is_genfloat<_Tp>::value>> {
public:
  typedef _Tp value_type;

private:
  value_type __re_;
  value_type __im_;

public:
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      value_type __re = value_type(), value_type __im = value_type())
      : __re_(__re), __im_(__im) {}

  template <typename _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(const complex<_Xp> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}

  template <class _Xp, typename = std::enable_if_t<is_genfloat<_Xp>::value>>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      const std::complex<_Xp> &__c)
      : __re_(static_cast<value_type>(__c.real())),
        __im_(static_cast<value_type>(__c.imag())) {}

  template <class _Xp, typename = std::enable_if_t<is_genfloat<_Xp>::value>>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
  operator std::complex<_Xp>() const {
    return std::complex<_Xp>(static_cast<_Xp>(__re_), static_cast<_Xp>(__im_));
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(value_type __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator+=(complex<value_type> &__c, value_type __re) {
    __c.__re_ += __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator-=(complex<value_type> &__c, value_type __re) {
    __c.__re_ -= __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator*=(complex<value_type> &__c, value_type __re) {
    __c.__re_ *= __re;
    __c.__im_ *= __re;
    return __c;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator/=(complex<value_type> &__c, value_type __re) {
    __c.__re_ /= __re;
    __c.__im_ /= __re;
    return __c;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator+=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x.__re_ += __y.real();
    __x.__im_ += __y.imag();
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator-=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x.__re_ -= __y.real();
    __x.__im_ -= __y.imag();
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator*=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x = __x * complex(__y.real(), __y.imag());
    return __x;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex &
  operator/=(complex<value_type> &__x, const complex<_Xp> &__y) {
    __x = __x / complex(__y.real(), __y.imag());
    return __x;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator+(const complex<value_type> &__x, const complex<value_type> &__y) {
    complex<value_type> __t(__x);
    __t += __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator+(const complex<value_type> &__x, value_type __y) {
    complex<value_type> __t(__x);
    __t += __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator+(value_type __x, const complex<value_type> &__y) {
    complex<value_type> __t(__y);
    __t += __x;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator+(const complex<value_type> &__x) {
    return __x;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator-(const complex<value_type> &__x, const complex<value_type> &__y) {
    complex<value_type> __t(__x);
    __t -= __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator-(const complex<value_type> &__x, value_type __y) {
    complex<value_type> __t(__x);
    __t -= __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator-(value_type __x, const complex<value_type> &__y) {
    complex<value_type> __t(-__y);
    __t += __x;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator-(const complex<value_type> &__x) {
    return complex<value_type>(-__x.__re_, -__x.__im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator*(const complex<value_type> &__z, const complex<value_type> &__w) {
    value_type __a = __z.__re_;
    value_type __b = __z.__im_;
    value_type __c = __w.__re_;
    value_type __d = __w.__im_;
    value_type __ac = __a * __c;
    value_type __bd = __b * __d;
    value_type __ad = __a * __d;
    value_type __bc = __b * __c;
    value_type __x = __ac - __bd;
    value_type __y = __ad + __bc;
    if (sycl::isnan(__x) && sycl::isnan(__y)) {
      bool __recalc = false;
      if (sycl::isinf(__a) || sycl::isinf(__b)) {
        __a = sycl::copysign(sycl::isinf(__a) ? value_type(1) : value_type(0),
                             __a);
        __b = sycl::copysign(sycl::isinf(__b) ? value_type(1) : value_type(0),
                             __b);
        if (sycl::isnan(__c))
          __c = sycl::copysign(value_type(0), __c);
        if (sycl::isnan(__d))
          __d = sycl::copysign(value_type(0), __d);
        __recalc = true;
      }
      if (sycl::isinf(__c) || sycl::isinf(__d)) {
        __c = sycl::copysign(sycl::isinf(__c) ? value_type(1) : value_type(0),
                             __c);
        __d = sycl::copysign(sycl::isinf(__d) ? value_type(1) : value_type(0),
                             __d);
        if (sycl::isnan(__a))
          __a = sycl::copysign(value_type(0), __a);
        if (sycl::isnan(__b))
          __b = sycl::copysign(value_type(0), __b);
        __recalc = true;
      }
      if (!__recalc && (sycl::isinf(__ac) || sycl::isinf(__bd) ||
                        sycl::isinf(__ad) || sycl::isinf(__bc))) {
        if (sycl::isnan(__a))
          __a = sycl::copysign(value_type(0), __a);
        if (sycl::isnan(__b))
          __b = sycl::copysign(value_type(0), __b);
        if (sycl::isnan(__c))
          __c = sycl::copysign(value_type(0), __c);
        if (sycl::isnan(__d))
          __d = sycl::copysign(value_type(0), __d);
        __recalc = true;
      }
      if (__recalc) {
        __x = value_type(INFINITY) * (__a * __c - __b * __d);
        __y = value_type(INFINITY) * (__a * __d + __b * __c);
      }
    }
    return complex<value_type>(__x, __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator*(const complex<value_type> &__x, value_type __y) {
    complex<value_type> __t(__x);
    __t *= __y;
    return __t;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator*(value_type __x, const complex<value_type> &__y) {
    complex<value_type> __t(__y);
    __t *= __x;
    return __t;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator/(const complex<value_type> &__z, const complex<value_type> &__w) {
    int __ilogbw = 0;
    value_type __a = __z.__re_;
    value_type __b = __z.__im_;
    value_type __c = __w.__re_;
    value_type __d = __w.__im_;
    value_type __logbw =
        sycl::logb(sycl::fmax(sycl::fabs(__c), sycl::fabs(__d)));
    if (sycl::isfinite(__logbw)) {
      __ilogbw = static_cast<int>(__logbw);
      __c = sycl::ldexp(__c, -__ilogbw);
      __d = sycl::ldexp(__d, -__ilogbw);
    }
    value_type __denom = __c * __c + __d * __d;
    value_type __x = sycl::ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
    value_type __y = sycl::ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (sycl::isnan(__x) && sycl::isnan(__y)) {
      if ((__denom == value_type(0)) &&
          (!sycl::isnan(__a) || !sycl::isnan(__b))) {
        __x = sycl::copysign(value_type(INFINITY), __c) * __a;
        __y = sycl::copysign(value_type(INFINITY), __c) * __b;
      } else if ((sycl::isinf(__a) || sycl::isinf(__b)) &&
                 sycl::isfinite(__c) && sycl::isfinite(__d)) {
        __a = sycl::copysign(sycl::isinf(__a) ? value_type(1) : value_type(0),
                             __a);
        __b = sycl::copysign(sycl::isinf(__b) ? value_type(1) : value_type(0),
                             __b);
        __x = value_type(INFINITY) * (__a * __c + __b * __d);
        __y = value_type(INFINITY) * (__b * __c - __a * __d);
      } else if (sycl::isinf(__logbw) && __logbw > value_type(0) &&
                 sycl::isfinite(__a) && sycl::isfinite(__b)) {
        __c = sycl::copysign(sycl::isinf(__c) ? value_type(1) : value_type(0),
                             __c);
        __d = sycl::copysign(sycl::isinf(__d) ? value_type(1) : value_type(0),
                             __d);
        __x = value_type(0) * (__a * __c + __b * __d);
        __y = value_type(0) * (__b * __c - __a * __d);
      }
    }
    return complex<value_type>(__x, __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator/(const complex<value_type> &__x, value_type __y) {
    return complex<value_type>(__x.__re_ / __y, __x.__im_ / __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend complex<value_type>
  operator/(value_type __x, const complex<value_type> &__y) {
    complex<value_type> __t(__x);
    __t /= __y;
    return __t;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator==(const complex<value_type> &__x, const complex<value_type> &__y) {
    return __x.__re_ == __y.__re_ && __x.__im_ == __y.__im_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator==(const complex<value_type> &__x, value_type __y) {
    return __x.__re_ == __y && __x.__im_ == 0;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator==(value_type __x, const complex<value_type> &__y) {
    return __x == __y.__re_ && 0 == __y.__im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator!=(const complex<value_type> &__x, const complex<value_type> &__y) {
    return !(__x == __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator!=(const complex<value_type> &__x, value_type __y) {
    return !(__x == __y);
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend constexpr bool
  operator!=(value_type __x, const complex<value_type> &__y) {
    return !(__x == __y);
  }

  template <class _CharT, class _Traits>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend std::basic_istream<_CharT, _Traits> &
  operator>>(std::basic_istream<_CharT, _Traits> &__is,
             complex<value_type> &__x) {
    if (__is.good()) {
      ws(__is);
      if (__is.peek() == _CharT('(')) {
        __is.get();
        value_type __r;
        __is >> __r;
        if (!__is.fail()) {
          ws(__is);
          _CharT __c = __is.peek();
          if (__c == _CharT(',')) {
            __is.get();
            value_type __i;
            __is >> __i;
            if (!__is.fail()) {
              ws(__is);
              __c = __is.peek();
              if (__c == _CharT(')')) {
                __is.get();
                __x = complex<value_type>(__r, __i);
              } else
                __is.setstate(__is.failbit);
            } else
              __is.setstate(__is.failbit);
          } else if (__c == _CharT(')')) {
            __is.get();
            __x = complex<value_type>(__r, value_type(0));
          } else
            __is.setstate(__is.failbit);
        } else
          __is.setstate(__is.failbit);
      } else {
        value_type __r;
        __is >> __r;
        if (!__is.fail())
          __x = complex<value_type>(__r, value_type(0));
        else
          __is.setstate(__is.failbit);
      }
    } else
      __is.setstate(__is.failbit);
    return __is;
  }

  template <class _CharT, class _Traits>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend std::basic_ostream<_CharT, _Traits> &
  operator<<(std::basic_ostream<_CharT, _Traits> &__os,
             const complex<value_type> &__x) {
    std::basic_ostringstream<_CharT, _Traits> __s;
    __s.flags(__os.flags());
    __s.imbue(__os.getloc());
    __s.precision(__os.precision());
    __s << '(' << __x.__re_ << ',' << __x.__im_ << ')';
    return __os << __s.str();
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY friend const sycl::stream &
  operator<<(const sycl::stream &__ss, const complex<value_type> &_x) {
    return __ss << "(" << _x.__re_ << "," << _x.__im_ << ")";
  }
};

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace _V1
} // namespace sycl
