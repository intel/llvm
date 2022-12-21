// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Adapted from the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef SYCL_EXT_ONEAPI_COMPLEX

#define _SYCL_EXT_CPLX_INLINE_VISIBILITY                                       \
  inline __attribute__((__visibility__("hidden"), __always_inline__))

#include <complex>
#include <sstream> // for std::basic_ostringstream
#include <sycl/sycl.hpp>
#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

using std::enable_if;
using std::integral_constant;
using std::is_floating_point;
using std::is_integral;
using std::is_same;

using std::basic_istream;
using std::basic_ostream;
using std::basic_ostringstream;

using std::declval;

template <bool _Val> using _BoolConstant = integral_constant<bool, _Val>;

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

  typedef decltype(__test(declval<_Tp>())) type;
  static const bool value = _IsNotSame<type, void>::value;
};

template <> struct __numeric_type<void> {
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&__numeric_type<_A2>::value
              &&__numeric_type<_A3>::value>
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

template <class _Tp> class complex;

template <class _Tp>
struct is_gencomplex
    : std::integral_constant<bool,
                             std::is_same_v<_Tp, complex<double>> ||
                                 std::is_same_v<_Tp, complex<float>> ||
                                 std::is_same_v<_Tp, complex<sycl::half>>> {};

template <class _Tp>
struct is_genfloat
    : std::integral_constant<bool, std::is_same_v<_Tp, double> ||
                                       std::is_same_v<_Tp, float> ||
                                       std::is_same_v<_Tp, sycl::half>> {};

template <class _Tp>
complex<_Tp> operator*(const complex<_Tp> &__z, const complex<_Tp> &__w);
template <class _Tp>
complex<_Tp> operator/(const complex<_Tp> &__x, const complex<_Tp> &__y);

template <class _Tp> class complex {
public:
  typedef _Tp value_type;

private:
  value_type __re_;
  value_type __im_;

public:
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      const value_type &__re = value_type(),
      const value_type &__im = value_type())
      : __re_(__re), __im_(__im) {}
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(const complex<_Xp> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      const std::complex<_Xp> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY operator std::complex<_Xp>() {
    return std::complex<_Xp>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const value_type &__re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator+=(const value_type &__re) {
    __re_ += __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator-=(const value_type &__re) {
    __re_ -= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator*=(const value_type &__re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator/=(const value_type &__re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator=(const std::complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator+=(const complex<_Xp> &__c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator-=(const complex<_Xp> &__c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator*=(const complex<_Xp> &__c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator/=(const complex<_Xp> &__c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

template <> class complex<float>;
template <> class complex<double>;

template <> class complex<sycl::half> {
  sycl::half __re_;
  sycl::half __im_;

public:
  typedef sycl::half value_type;

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      sycl::half __re = sycl::half{}, sycl::half __im = sycl::half{})
      : __re_(__re), __im_(__im) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  explicit constexpr complex(const complex<float> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  explicit constexpr complex(const complex<double> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr operator std::complex<sycl::half>() {
    return std::complex<sycl::half>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr sycl::half real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr sycl::half imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(sycl::half __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator+=(sycl::half __re) {
    __re_ += __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator-=(sycl::half __re) {
    __re_ -= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator*=(sycl::half __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator/=(sycl::half __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator=(const std::complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator+=(const complex<_Xp> &__c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator-=(const complex<_Xp> &__c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator*=(const complex<_Xp> &__c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator/=(const complex<_Xp> &__c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

template <> class complex<float> {
  float __re_;
  float __im_;

public:
  typedef float value_type;

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(float __re = 0.0f,
                                                     float __im = 0.0f)
      : __re_(__re), __im_(__im) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const complex<sycl::half> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  explicit constexpr complex(const complex<double> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const std::complex<float> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr operator std::complex<float>() {
    return std::complex<float>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr float real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr float imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(float __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator+=(float __re) {
    __re_ += __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator-=(float __re) {
    __re_ -= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator*=(float __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator/=(float __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator=(const std::complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator+=(const complex<_Xp> &__c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator-=(const complex<_Xp> &__c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator*=(const complex<_Xp> &__c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator/=(const complex<_Xp> &__c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

template <> class complex<double> {
  double __re_;
  double __im_;

public:
  typedef double value_type;

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(double __re = 0.0,
                                                     double __im = 0.0)
      : __re_(__re), __im_(__im) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const complex<sycl::half> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const complex<float> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const std::complex<double> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr operator std::complex<double>() {
    return std::complex<double>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr double real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr double imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(double __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator+=(double __re) {
    __re_ += __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator-=(double __re) {
    __re_ -= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator*=(double __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator/=(double __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator=(const std::complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator+=(const complex<_Xp> &__c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator-=(const complex<_Xp> &__c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator*=(const complex<_Xp> &__c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator/=(const complex<_Xp> &__c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

inline constexpr complex<sycl::half>::complex(const complex<float> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<sycl::half>::complex(const complex<double> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<float>::complex(const complex<sycl::half> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<float>::complex(const complex<double> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<double>::complex(const complex<sycl::half> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<double>::complex(const complex<float> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

// 26.3.6 operators:

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator+(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__x);
  __t += __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> operator+(const complex<_Tp> &__x,
                                                        const _Tp &__y) {
  complex<_Tp> __t(__x);
  __t += __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator+(const _Tp &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__y);
  __t += __x;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator-(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__x);
  __t -= __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> operator-(const complex<_Tp> &__x,
                                                        const _Tp &__y) {
  complex<_Tp> __t(__x);
  __t -= __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator-(const _Tp &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(-__y);
  __t += __x;
  return __t;
}

template <class _Tp>
complex<_Tp> operator*(const complex<_Tp> &__z, const complex<_Tp> &__w) {
  _Tp __a = __z.real();
  _Tp __b = __z.imag();
  _Tp __c = __w.real();
  _Tp __d = __w.imag();
  _Tp __ac = __a * __c;
  _Tp __bd = __b * __d;
  _Tp __ad = __a * __d;
  _Tp __bc = __b * __c;
  _Tp __x = __ac - __bd;
  _Tp __y = __ad + __bc;
  if (sycl::isnan(__x) && sycl::isnan(__y)) {
    bool __recalc = false;
    if (sycl::isinf(__a) || sycl::isinf(__b)) {
      __a = sycl::copysign(sycl::isinf(__a) ? _Tp(1) : _Tp(0), __a);
      __b = sycl::copysign(sycl::isinf(__b) ? _Tp(1) : _Tp(0), __b);
      if (sycl::isnan(__c))
        __c = sycl::copysign(_Tp(0), __c);
      if (sycl::isnan(__d))
        __d = sycl::copysign(_Tp(0), __d);
      __recalc = true;
    }
    if (sycl::isinf(__c) || sycl::isinf(__d)) {
      __c = sycl::copysign(sycl::isinf(__c) ? _Tp(1) : _Tp(0), __c);
      __d = sycl::copysign(sycl::isinf(__d) ? _Tp(1) : _Tp(0), __d);
      if (sycl::isnan(__a))
        __a = sycl::copysign(_Tp(0), __a);
      if (sycl::isnan(__b))
        __b = sycl::copysign(_Tp(0), __b);
      __recalc = true;
    }
    if (!__recalc && (sycl::isinf(__ac) || sycl::isinf(__bd) ||
                      sycl::isinf(__ad) || sycl::isinf(__bc))) {
      if (sycl::isnan(__a))
        __a = sycl::copysign(_Tp(0), __a);
      if (sycl::isnan(__b))
        __b = sycl::copysign(_Tp(0), __b);
      if (sycl::isnan(__c))
        __c = sycl::copysign(_Tp(0), __c);
      if (sycl::isnan(__d))
        __d = sycl::copysign(_Tp(0), __d);
      __recalc = true;
    }
    if (__recalc) {
      __x = _Tp(INFINITY) * (__a * __c - __b * __d);
      __y = _Tp(INFINITY) * (__a * __d + __b * __c);
    }
  }
  return complex<_Tp>(__x, __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> operator*(const complex<_Tp> &__x,
                                                        const _Tp &__y) {
  complex<_Tp> __t(__x);
  __t *= __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator*(const _Tp &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__y);
  __t *= __x;
  return __t;
}

template <class _Tp>
complex<_Tp> operator/(const complex<_Tp> &__z, const complex<_Tp> &__w) {
  int __ilogbw = 0;
  _Tp __a = __z.real();
  _Tp __b = __z.imag();
  _Tp __c = __w.real();
  _Tp __d = __w.imag();
  _Tp __logbw = sycl::logb(sycl::fmax(sycl::fabs(__c), sycl::fabs(__d)));
  if (sycl::isfinite(__logbw)) {
    __ilogbw = static_cast<int>(__logbw);
    __c = sycl::ldexp(__c, -__ilogbw);
    __d = sycl::ldexp(__d, -__ilogbw);
  }
  _Tp __denom = __c * __c + __d * __d;
  _Tp __x = sycl::ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
  _Tp __y = sycl::ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (sycl::isnan(__x) && sycl::isnan(__y)) {
    if ((__denom == _Tp(0)) && (!sycl::isnan(__a) || !sycl::isnan(__b))) {
      __x = sycl::copysign(_Tp(INFINITY), __c) * __a;
      __y = sycl::copysign(_Tp(INFINITY), __c) * __b;
    } else if ((sycl::isinf(__a) || sycl::isinf(__b)) && sycl::isfinite(__c) &&
               sycl::isfinite(__d)) {
      __a = sycl::copysign(sycl::isinf(__a) ? _Tp(1) : _Tp(0), __a);
      __b = sycl::copysign(sycl::isinf(__b) ? _Tp(1) : _Tp(0), __b);
      __x = _Tp(INFINITY) * (__a * __c + __b * __d);
      __y = _Tp(INFINITY) * (__b * __c - __a * __d);
    } else if (sycl::isinf(__logbw) && __logbw > _Tp(0) &&
               sycl::isfinite(__a) && sycl::isfinite(__b)) {
      __c = sycl::copysign(sycl::isinf(__c) ? _Tp(1) : _Tp(0), __c);
      __d = sycl::copysign(sycl::isinf(__d) ? _Tp(1) : _Tp(0), __d);
      __x = _Tp(0) * (__a * __c + __b * __d);
      __y = _Tp(0) * (__b * __c - __a * __d);
    }
  }
  return complex<_Tp>(__x, __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> operator/(const complex<_Tp> &__x,
                                                        const _Tp &__y) {
  return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator/(const _Tp &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__x);
  __t /= __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator+(const complex<_Tp> &__x) {
  return __x;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator-(const complex<_Tp> &__x) {
  return complex<_Tp>(-__x.real(), -__x.imag());
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator==(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator==(const complex<_Tp> &__x, const _Tp &__y) {
  return __x.real() == __y && __x.imag() == 0;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator==(const _Tp &__x, const complex<_Tp> &__y) {
  return __x == __y.real() && 0 == __y.imag();
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator!=(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  return !(__x == __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator!=(const complex<_Tp> &__x, const _Tp &__y) {
  return !(__x == __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator!=(const _Tp &__x, const complex<_Tp> &__y) {
  return !(__x == __y);
}

// 26.3.7 values:

template <class _Tp, bool = is_integral<_Tp>::value,
          bool = is_floating_point<_Tp>::value>
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

// real

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _Tp
real(const complex<_Tp> &__c) {
  return __c.real();
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename __libcpp_complex_overload_traits<_Tp>::_ValueType
    real(_Tp __re) {
  return __re;
}

// imag

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _Tp
imag(const complex<_Tp> &__c) {
  return __c.imag();
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename __libcpp_complex_overload_traits<_Tp>::_ValueType
    imag(_Tp) {
  return 0;
}

// abs

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp
abs(const complex<_Tp> &__c) {
  return sycl::hypot(__c.real(), __c.imag());
}

// arg

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp
arg(const complex<_Tp> &__c) {
  return sycl::atan2(__c.imag(), __c.real());
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename enable_if<is_integral<_Tp>::value || is_same<_Tp, double>::value,
                       double>::type
    arg(_Tp __re) {
  return sycl::atan2(0., __re);
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename enable_if<is_same<_Tp, float>::value, float>::type
    arg(_Tp __re) {
  return sycl::atan2(0.F, __re);
}

// norm

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp
norm(const complex<_Tp> &__c) {
  if (sycl::isinf(__c.real()))
    return sycl::fabs(__c.real());
  if (sycl::isinf(__c.imag()))
    return sycl::fabs(__c.imag());
  return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename __libcpp_complex_overload_traits<_Tp>::_ValueType
    norm(_Tp __re) {
  typedef typename __libcpp_complex_overload_traits<_Tp>::_ValueType _ValueType;
  return static_cast<_ValueType>(__re) * __re;
}

// conj

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
conj(const complex<_Tp> &__c) {
  return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
    conj(_Tp __re) {
  typedef
      typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
  return _ComplexType(__re);
}

// proj

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
proj(const complex<_Tp> &__c) {
  complex<_Tp> __r = __c;
  if (sycl::isinf(__c.real()) || sycl::isinf(__c.imag()))
    __r = complex<_Tp>(INFINITY, sycl::copysign(_Tp(0), __c.imag()));
  return __r;
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY typename enable_if<
    is_floating_point<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType>::type
proj(_Tp __re) {
  if (sycl::isinf(__re))
    __re = sycl::fabs(__re);
  return complex<_Tp>(__re);
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY typename enable_if<
    is_integral<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType>::type
proj(_Tp __re) {
  typedef
      typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
  return _ComplexType(__re);
}

// polar

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> polar(const _Tp &__rho, const _Tp &__theta = _Tp()) {
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

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
log(const complex<_Tp> &__x) {
  return complex<_Tp>(sycl::log(abs(__x)), arg(__x));
}

// log10

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
log10(const complex<_Tp> &__x) {
  return log(__x) / sycl::log(_Tp(10));
}

// sqrt

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> sqrt(const complex<_Tp> &__x) {
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

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> exp(const complex<_Tp> &__x) {
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

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
pow(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  return exp(__y * log(__x));
}

template <class _Tp, class _Up,
          class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL
    _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<typename __promote<_Tp, _Up>::type>
    pow(const complex<_Tp> &__x, const complex<_Up> &__y) {
  typedef complex<typename __promote<_Tp, _Up>::type> result_type;
  return sycl::ext::oneapi::experimental::pow(result_type(__x),
                                              result_type(__y));
}

template <class _Tp, class _Up,
          class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename enable_if<is_genfloat<_Up>::value,
                       complex<typename __promote<_Tp, _Up>::type>>::type
    pow(const complex<_Tp> &__x, const _Up &__y) {
  typedef complex<typename __promote<_Tp, _Up>::type> result_type;
  return sycl::ext::oneapi::experimental::pow(result_type(__x),
                                              result_type(__y));
}

template <class _Tp, class _Up,
          class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY
    typename enable_if<is_genfloat<_Up>::value,
                       complex<typename __promote<_Tp, _Up>::type>>::type
    pow(const _Tp &__x, const complex<_Up> &__y) {
  typedef complex<typename __promote<_Tp, _Up>::type> result_type;
  return sycl::ext::oneapi::experimental::pow(result_type(__x),
                                              result_type(__y));
}

// __sqr, computes pow(x, 2)

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> __sqr(const complex<_Tp> &__x) {
  return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()),
                      _Tp(2) * __x.real() * __x.imag());
}

// asinh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> asinh(const complex<_Tp> &__x) {
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
  complex<_Tp> __z = log(__x + sqrt(__sqr(__x) + _Tp(1)));
  return complex<_Tp>(sycl::copysign(__z.real(), __x.real()),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// acosh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> acosh(const complex<_Tp> &__x) {
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
  complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
  return complex<_Tp>(sycl::copysign(__z.real(), _Tp(0)),
                      sycl::copysign(__z.imag(), __x.imag()));
}

// atanh

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> atanh(const complex<_Tp> &__x) {
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

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> sinh(const complex<_Tp> &__x) {
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

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> cosh(const complex<_Tp> &__x) {
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

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> tanh(const complex<_Tp> &__x) {
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

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> asin(const complex<_Tp> &__x) {
  complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> acos(const complex<_Tp> &__x) {
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
  complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
  if (sycl::signbit(__x.imag()))
    return complex<_Tp>(sycl::fabs(__z.imag()), sycl::fabs(__z.real()));
  return complex<_Tp>(sycl::fabs(__z.imag()), -sycl::fabs(__z.real()));
}

// atan

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
atan(const complex<_Tp> &__x) {
  complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
sin(const complex<_Tp> &__x) {
  complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
cos(const complex<_Tp> &__x) {
  return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
tan(const complex<_Tp> &__x) {
  complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

template <class _Tp, class _CharT, class _Traits>
basic_istream<_CharT, _Traits> &operator>>(basic_istream<_CharT, _Traits> &__is,
                                           complex<_Tp> &__x) {
  if (__is.good()) {
    ws(__is);
    if (__is.peek() == _CharT('(')) {
      __is.get();
      _Tp __r;
      __is >> __r;
      if (!__is.fail()) {
        ws(__is);
        _CharT __c = __is.peek();
        if (__c == _CharT(',')) {
          __is.get();
          _Tp __i;
          __is >> __i;
          if (!__is.fail()) {
            ws(__is);
            __c = __is.peek();
            if (__c == _CharT(')')) {
              __is.get();
              __x = complex<_Tp>(__r, __i);
            } else
              __is.setstate(__is.failbit);
          } else
            __is.setstate(__is.failbit);
        } else if (__c == _CharT(')')) {
          __is.get();
          __x = complex<_Tp>(__r, _Tp(0));
        } else
          __is.setstate(__is.failbit);
      } else
        __is.setstate(__is.failbit);
    } else {
      _Tp __r;
      __is >> __r;
      if (!__is.fail())
        __x = complex<_Tp>(__r, _Tp(0));
      else
        __is.setstate(__is.failbit);
    }
  } else
    __is.setstate(__is.failbit);
  return __is;
}

template <class _Tp, class _CharT, class _Traits>
basic_ostream<_CharT, _Traits> &operator<<(basic_ostream<_CharT, _Traits> &__os,
                                           const complex<_Tp> &__x) {
  basic_ostringstream<_CharT, _Traits> __s;
  __s.flags(__os.flags());
  __s.imbue(__os.getloc());
  __s.precision(__os.precision());
  __s << '(' << __x.real() << ',' << __x.imag() << ')';
  return __os << __s.str();
}

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY inline const sycl::stream &
operator<<(const sycl::stream &__ss, const complex<_Tp> &_x) {
  return __ss << "(" << _x.real() << "," << _x.imag() << ")";
}

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

#undef _SYCL_EXT_CPLX_INLINE_VISIBILITY

#endif // SYCL_EXT_ONEAPI_COMPLEX
