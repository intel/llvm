//===- sycl_complex.hpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef SYCL_EXT_ONEAPI_COMPLEX

#define _SYCL_EXT_CPLX_INLINE_VISIBILITY                                       \
  inline __attribute__((__visibility__("hidden"), __always_inline__))

#include <sycl/detail/builtins.hpp> // for isinf, isnan...
#include <sycl/group_algorithm.hpp> // for reduce_over_group...
#include <sycl/stream.hpp>          // for stream...

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

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

} // namespace cplx::detail

////////////////////////////////////////////////////////////////////////////////
/// COMPLEX
////////////////////////////////////////////////////////////////////////////////

template <class _Tp, class _Enable = void> class complex;

template <class _Tp>
struct is_gencomplex
    : std::integral_constant<bool,
                             std::is_same_v<_Tp, complex<double>> ||
                                 std::is_same_v<_Tp, complex<float>> ||
                                 std::is_same_v<_Tp, complex<sycl::half>>> {};
template <typename _Tp>
inline constexpr bool is_gencomplex_v = is_gencomplex<_Tp>::value;

template <class _Tp>
struct is_genfloat
    : std::integral_constant<bool, std::is_same_v<_Tp, double> ||
                                       std::is_same_v<_Tp, float> ||
                                       std::is_same_v<_Tp, sycl::half>> {};
template <typename _Tp>
inline constexpr bool is_genfloat_v = is_genfloat<_Tp>::value;

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

////////////////////////////////////////////////////////////////////////////////
/// MATH COMPLEX FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

namespace cplx::detail {

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
  return sycl::atan2(static_cast<_ValueType>(0), __re);
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

} // namespace experimental
} // namespace oneapi
} // namespace ext

////////////////////////////////////////////////////////////////////////////////
/// MARRAY
////////////////////////////////////////////////////////////////////////////////

namespace ext {
namespace oneapi {
namespace experimental {

template <typename T> using marray_data = sycl::detail::vec_helper<T>;

template <typename T>
using marray_data_t = typename sycl::detail::vec_helper<T>::RetType;

template <typename T> struct is_mgencomplex : std::false_type {};
template <typename T, std::size_t N>
struct is_mgencomplex<sycl::marray<T, N>>
    : std::integral_constant<
          bool, sycl::ext::oneapi::experimental::is_gencomplex_v<T>> {};
template <typename T>
inline constexpr bool is_mgencomplex_v = is_mgencomplex<T>::value;

} // namespace experimental
} // namespace oneapi
} // namespace ext

template <typename T, std::size_t NumElements>
class marray<sycl::ext::oneapi::experimental::complex<T>, NumElements> {
private:
  using ComplexDataT = sycl::ext::oneapi::experimental::complex<T>;

public:
  using value_type = ComplexDataT;
  using reference = ComplexDataT &;
  using const_reference = const ComplexDataT &;
  using iterator = ComplexDataT *;
  using const_iterator = const ComplexDataT *;

private:
  value_type MData[NumElements];

  template <size_t... Is>
  constexpr marray(const std::array<value_type, NumElements> &Arr,
                   std::index_sequence<Is...>)
      : MData{Arr[Is]...} {}

  /// FIXME: If the subscript operator is made constexpr this can be removed.
  // detail::HelperFlattenMArrayArg::MArrayToArray needs to have access to
  // MData.
  friend class detail::HelperFlattenMArrayArg;

public:
  constexpr marray() : MData{} {};

  explicit constexpr marray(const value_type &arg)
      : marray{
            sycl::detail::RepeatValue<NumElements>(
                static_cast<
                    ext::oneapi::experimental::marray_data_t<value_type>>(arg)),
            std::make_index_sequence<NumElements>()} {}

  template <
      typename... ArgTN,
      typename = std::enable_if_t<
          sycl::detail::AllSuitableArgTypes<value_type, ArgTN...>::value &&
          sycl::detail::GetMArrayArgsSize<ArgTN...>::value == NumElements>>
  constexpr marray(const ArgTN &...Args)
      : marray{
            sycl::detail::MArrayArgArrayCreator<value_type, ArgTN...>::Create(
                Args...),
            std::make_index_sequence<NumElements>()} {}

  constexpr marray(const marray<value_type, NumElements> &rhs) = default;
  constexpr marray(marray<value_type, NumElements> &&rhs) = default;

  // Available only when: NumElements == 1
  template <std::size_t N = NumElements,
            typename = std::enable_if_t<N == 1, value_type>>
  operator value_type() const {
    return MData[0];
  }

  static constexpr std::size_t size() noexcept { return NumElements; }

  marray<T, NumElements> real() const {
    sycl::marray<T, NumElements> rtn;
    for (std::size_t i = 0; i < NumElements; ++i) {
      rtn[i] = MData[i].real();
    }
    return rtn;
  }

  marray<T, NumElements> imag() const {
    sycl::marray<T, NumElements> rtn;
    for (std::size_t i = 0; i < NumElements; ++i) {
      rtn[i] = MData[i].imag();
    }
    return rtn;
  }

  // subscript operator
  reference operator[](std::size_t i) { return MData[i]; }
  const_reference operator[](std::size_t i) const { return MData[i]; }

  marray &operator=(const marray<value_type, NumElements> &rhs) = default;
  marray &operator=(const value_type &rhs) {
    for (std::size_t i = 0; i < NumElements; ++i) {
      MData[i] = rhs;
    }
    return *this;
  }

  // iterator functions
  iterator begin() { return MData; }
  const_iterator begin() const { return MData; }

  iterator end() { return MData + NumElements; }
  const_iterator end() const { return MData + NumElements; }

#ifdef MARRAY_CPLX_OP
#error "Multiple definition of MARRAY_CPLX_OP"
#endif

  // MARRAY_CPLX_OP is: +, -, *, /
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray operator op(const marray &lhs, const marray &rhs) {            \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs[i] op rhs[i];                                               \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray operator op(const marray &lhs, const value_type &rhs) {        \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs[i] op rhs;                                                  \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray operator op(const value_type &lhs, const marray &rhs) {        \
    marray rtn;                                                                \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs op rhs[i];                                                  \
    }                                                                          \
    return rtn;                                                                \
  }

  MARRAY_CPLX_OP(+)
  MARRAY_CPLX_OP(-)
  MARRAY_CPLX_OP(*)
  MARRAY_CPLX_OP(/)

#undef MARRAY_CPLX_OP

  // MARRAY_CPLX_OP is: %
  friend marray operator%(const marray &lhs, const marray &rhs) = delete;
  friend marray operator%(const marray &lhs, const value_type &rhs) = delete;
  friend marray operator%(const value_type &lhs, const marray &rhs) = delete;

  // MARRAY_CPLX_OP is: +=, -=, *=, /=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray &operator op(marray &lhs, const marray &rhs) {                 \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      lhs[i] op rhs[i];                                                        \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
                                                                               \
  friend marray &operator op(marray &lhs, const value_type &rhs) {             \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      lhs[i] op rhs;                                                           \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
  friend marray &operator op(value_type &lhs, const marray &rhs) {             \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      lhs[i] op rhs;                                                           \
    }                                                                          \
    return lhs;                                                                \
  }

  MARRAY_CPLX_OP(+=)
  MARRAY_CPLX_OP(-=)
  MARRAY_CPLX_OP(*=)
  MARRAY_CPLX_OP(/=)

#undef MARRAY_CPLX_OP

  // MARRAY_CPLX_OP is: %=
  friend marray &operator%=(marray &lhs, const marray &rhs) = delete;
  friend marray &operator%=(marray &lhs, const value_type &rhs) = delete;
  friend marray &operator%=(value_type &lhs, const marray &rhs) = delete;

// MARRAY_CPLX_OP is: ++, --
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray operator op(marray &lhs, int) = delete;                        \
  friend marray &operator op(marray &rhs) = delete;

  MARRAY_CPLX_OP(++)
  MARRAY_CPLX_OP(--)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: unary +, unary -
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray<value_type, NumElements> operator op(                          \
      const marray<value_type, NumElements> &rhs) {                            \
    marray<value_type, NumElements> rtn;                                       \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = op rhs[i];                                                      \
    }                                                                          \
    return rtn;                                                                \
  }

  MARRAY_CPLX_OP(+)
  MARRAY_CPLX_OP(-)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: &, |, ^
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray operator op(const marray &lhs, const marray &rhs) = delete;    \
  friend marray operator op(const marray &lhs, const value_type &rhs) = delete;

  MARRAY_CPLX_OP(&)
  MARRAY_CPLX_OP(|)
  MARRAY_CPLX_OP(^)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: &=, |=, ^=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray &operator op(marray &lhs, const marray &rhs) = delete;         \
  friend marray &operator op(marray &lhs, const value_type &rhs) = delete;     \
  friend marray &operator op(value_type &lhs, const marray &rhs) = delete;

  MARRAY_CPLX_OP(&=)
  MARRAY_CPLX_OP(|=)
  MARRAY_CPLX_OP(^=)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: &&, ||
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const marray &rhs) = delete;    \
  friend marray<bool, NumElements> operator op(                                \
      const marray &lhs, const value_type &rhs) = delete;                      \
  friend marray<bool, NumElements> operator op(const value_type &lhs,          \
                                               const marray &rhs) = delete;

  MARRAY_CPLX_OP(&&)
  MARRAY_CPLX_OP(||)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: <<, >>
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray operator op(const marray &lhs, const marray &rhs) = delete;    \
  friend marray operator op(const marray &lhs, const value_type &rhs) =        \
      delete;                                                                  \
  friend marray operator op(const value_type &lhs, const marray &rhs) = delete;

  MARRAY_CPLX_OP(<<)
  MARRAY_CPLX_OP(>>)

#undef MARRAY_CPLX_OP

// MARRAY_CPLX_OP is: <<=, >>=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray &operator op(marray &lhs, const marray &rhs) = delete;         \
  friend marray &operator op(marray &lhs, const value_type &rhs) = delete;

  MARRAY_CPLX_OP(<<=)
  MARRAY_CPLX_OP(>>=)

#undef MARRAY_CPLX_OP

  // MARRAY_CPLX_OP is: ==, !=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const marray &rhs) {            \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs[i] op rhs[i];                                               \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const value_type &rhs) {        \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs[i] op rhs;                                                  \
    }                                                                          \
    return rtn;                                                                \
  }                                                                            \
                                                                               \
  friend marray<bool, NumElements> operator op(const value_type &lhs,          \
                                               const marray &rhs) {            \
    marray<bool, NumElements> rtn;                                             \
    for (std::size_t i = 0; i < NumElements; ++i) {                            \
      rtn[i] = lhs op rhs[i];                                                  \
    }                                                                          \
    return rtn;                                                                \
  }

  MARRAY_CPLX_OP(==)
  MARRAY_CPLX_OP(!=)

#undef MARRAY_CPLX_OP

  // MARRAY_CPLX_OP is: <, >, <=, >=
#define MARRAY_CPLX_OP(op)                                                     \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const marray &rhs) = delete;    \
  friend marray<bool, NumElements> operator op(                                \
      const marray &lhs, const value_type &rhs) = delete;                      \
  friend marray<bool, NumElements> operator op(const value_type &lhs,          \
                                               const marray &rhs) = delete;

  MARRAY_CPLX_OP(<);
  MARRAY_CPLX_OP(>);
  MARRAY_CPLX_OP(<=);
  MARRAY_CPLX_OP(>=);

#undef MARRAY_CPLX_OP

  friend marray operator~(const marray &v) = delete;

  friend marray<bool, NumElements> operator!(const marray &v) = delete;
};

////////////////////////////////////////////////////////////////////////////////
/// MARRAY COMPLEX MATH FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

namespace ext {
namespace oneapi {
namespace experimental {

// Math marray overloads

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

////////////////////////////////////////////////////////////////////////////////
/// COMPLEX && MARRAY COMPLEX GROUP ALGORITMHS FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

namespace cplx::detail {

/// Helper trait to check if the type is a sycl::plus
template <typename T> struct is_plus : std::false_type {};
template <typename T> struct is_plus<sycl::plus<T>> : std::true_type {};
template <typename BinaryOperation>
constexpr bool is_plus_v = is_plus<BinaryOperation>::value;

/// Helper trait to check if the type is a sycl:multiplies
template <typename T> struct is_multiplies : std::false_type {};
template <typename T>
struct is_multiplies<sycl::multiplies<T>> : std::true_type {};
template <typename BinaryOperation>
constexpr bool is_multiplies_v = is_multiplies<BinaryOperation>::value;

/// Wrapper trait to check if the binary operation is supported
template <typename BinaryOperation>
struct is_binary_op_supported
    : std::integral_constant<bool, (is_plus_v<BinaryOperation> ||
                                    is_multiplies_v<BinaryOperation>)> {};
template <class BinaryOperation>
inline constexpr bool is_binary_op_supported_v =
    is_binary_op_supported<BinaryOperation>::value;

/// Helper function to get the init for sycl::plus binary operation when the
/// type is a gencomplex
template <typename T, class BinaryOperation>
std::enable_if_t<(sycl::ext::oneapi::experimental::is_gencomplex_v<T> &&
                  is_plus_v<BinaryOperation>),
                 T>
get_init() {
  return T{0, 0};
}
/// Helper function to get the init for sycl::multiply binary operation when
/// the type is a gencomplex
template <typename T, class BinaryOperation>
std::enable_if_t<(sycl::ext::oneapi::experimental::is_gencomplex_v<T> &&
                  is_multiplies_v<BinaryOperation>),
                 T>
get_init() {
  return T{1, 0};
}
/// Helper function to get the init for sycl::plus binary operation when the
/// type is a mgencomplex
template <typename T, class BinaryOperation>
std::enable_if_t<(sycl::ext::oneapi::experimental::is_mgencomplex_v<T> &&
                  is_plus_v<BinaryOperation>),
                 T>
get_init() {
  using Complex = typename T::value_type;

  T result;
  std::fill(result.begin(), result.end(), Complex{0, 0});
  return result;
}
/// Helper function to get the init for sycl::multiply binary operation when
/// the type is a mgencomplex
template <typename T, class BinaryOperation>
std::enable_if_t<(sycl::ext::oneapi::experimental::is_mgencomplex_v<T> &&
                  is_multiplies_v<BinaryOperation>),
                 T>
get_init() {
  using Complex = typename T::value_type;

  T result;
  std::fill(result.begin(), result.end(), Complex{1, 0});
  return result;
}

/// Helper funtions to construct a __spv::complex_type from a sycl::complex
__spv::complex_half construct_spv_complex(sycl::half real, sycl::half imag) {
  return __spv::complex_half(real, imag);
}
__spv::complex_float construct_spv_complex(float real, float imag) {
  return __spv::complex_float(real, imag);
}
__spv::complex_double construct_spv_complex(double real, double imag) {
  return __spv::complex_double(real, imag);
}

} // namespace cplx::detail

/* REDUCE_OVER_GROUP'S OVERLOADS */

/// Complex specialization
template <typename Group, typename V, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_genfloat_v<V> &&
              is_genfloat_v<T> &&
              cplx::detail::is_binary_op_supported_v<BinaryOperation>>>
complex<T> reduce_over_group(Group g, complex<V> x, complex<T> init,
                             BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__

  complex<T> result;

  if constexpr (cplx::detail::is_plus_v<BinaryOperation>) {
    result.real(sycl::reduce_over_group(g, x.real(), init.real(), binary_op));
    result.imag(sycl::reduce_over_group(g, x.imag(), init.imag(), binary_op));
  } else {
    const auto flag = sycl::detail::spirv::group_scope<Group>::value;
    const auto operation =
        static_cast<unsigned int>(__spv::GroupOperation::Reduce);

    const auto spv_input =
        cplx::detail::construct_spv_complex(x.real(), x.imag());
    const auto spv_output = __spirv_GroupCMulINTEL(flag, operation, spv_input);

    result.real(spv_output.real);
    result.imag(spv_output.imag);
  }

  return result;

#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> specialization
template <typename Group, typename V, std::size_t N, typename T, std::size_t S,
          class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_gencomplex_v<V> &&
              is_gencomplex_v<T> &&
              cplx::detail::is_binary_op_supported_v<BinaryOperation>>>
sycl::marray<T, N> reduce_over_group(Group g, sycl::marray<V, N> x,
                                     sycl::marray<T, S> init,
                                     BinaryOperation binary_op) {
  sycl::marray<T, N> result;

  sycl::detail::loop<N>([&](size_t s) {
    result[s] = reduce_over_group(g, x[s], init[s], binary_op);
  });

  return result;
}

/// Marray<Complex> and Complex specialization
template <typename Group, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              (is_gencomplex_v<T> || is_mgencomplex_v<T>)&&cplx::detail::
                  is_binary_op_supported_v<BinaryOperation>>>
T reduce_over_group(Group g, T x, BinaryOperation binary_op) {
  auto init = cplx::detail::get_init<T, BinaryOperation>();

  return reduce_over_group(g, x, init, binary_op);
}

/* JOINT_REDUCE'S OVERLOADS */

/// Marray<Complex> and Complex specialization
template <typename Group, typename Ptr, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              sycl::detail::is_pointer<Ptr>::value &&
              (is_gencomplex_v<sycl::detail::remove_pointer_t<Ptr>> ||
               is_mgencomplex_v<sycl::detail::remove_pointer_t<
                   Ptr>>)&&(is_gencomplex_v<T> || is_mgencomplex_v<T>)&&cplx::
                  detail::is_binary_op_supported_v<BinaryOperation>>>
T joint_reduce(Group g, Ptr first, Ptr last, T init,
               BinaryOperation binary_op) {

  auto partial = cplx::detail::get_init<T, BinaryOperation>();

  sycl::detail::for_each(
      g, first, last,
      [&](const typename sycl::detail::remove_pointer<Ptr>::type &x) {
        partial = binary_op(partial, x);
      });

  return reduce_over_group(g, partial, init, binary_op);
}

/// Marray<Complex> and Complex specialization
template <typename Group, typename Ptr, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              sycl::detail::is_pointer<Ptr>::value &&
              (is_gencomplex_v<sycl::detail::remove_pointer_t<Ptr>> ||
               is_mgencomplex_v<sycl::detail::remove_pointer_t<Ptr>>)&&cplx::
                  detail::is_binary_op_supported_v<BinaryOperation>>>
typename sycl::detail::remove_pointer_t<Ptr>
joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {

  using T = typename sycl::detail::remove_pointer_t<Ptr>;

  auto init = cplx::detail::get_init<T, BinaryOperation>();

  return joint_reduce(g, first, last, init, binary_op);
}

/* INCLUSIVE_SCAN_OVER_GROUP'S OVERLOADS */

/// Complex specialization
template <typename Group, typename V, class BinaryOperation, typename T,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_genfloat_v<V> &&
              is_genfloat_v<T> &&
              cplx::detail::is_binary_op_supported_v<BinaryOperation>>>
complex<T> inclusive_scan_over_group(Group g, complex<V> x,
                                     BinaryOperation binary_op,
                                     complex<T> init) {
#ifdef __SYCL_DEVICE_ONLY__

  complex<T> result;

  if constexpr (cplx::detail::is_plus_v<BinaryOperation>) {
    result.real(
        sycl::inclusive_scan_over_group(g, x.real(), binary_op, init.real()));
    result.imag(
        sycl::inclusive_scan_over_group(g, x.imag(), binary_op, init.imag()));
  } else {
    const auto flag = sycl::detail::spirv::group_scope<Group>::value;
    const auto operation =
        static_cast<unsigned int>(__spv::GroupOperation::InclusiveScan);

    const auto spv_input =
        cplx::detail::construct_spv_complex(x.real(), x.imag());
    const auto spv_output = __spirv_GroupCMulINTEL(flag, operation, spv_input);

    result.real(spv_output.real);
    result.imag(spv_output.imag);
  }

  return result;
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> specialization
template <typename Group, typename V, std::size_t N, class BinaryOperation,
          typename T, std::size_t S,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_gencomplex_v<V> &&
              is_gencomplex_v<T> &&
              cplx::detail::is_binary_op_supported_v<BinaryOperation>>>
sycl::marray<T, N> inclusive_scan_over_group(Group g, sycl::marray<V, N> x,
                                             BinaryOperation binary_op,
                                             sycl::marray<T, S> init) {
  sycl::marray<T, N> result;

  sycl::detail::loop<N>([&](size_t s) {
    result[s] = inclusive_scan_over_group(g, x[s], binary_op, init[s]);
  });

  return result;
}

/// Marray<Complex> and Complex specialization
template <typename Group, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              (is_gencomplex_v<T> || is_mgencomplex_v<T>)&&cplx::detail::
                  is_binary_op_supported_v<BinaryOperation>>>
T inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  auto init = cplx::detail::get_init<T, BinaryOperation>();

  return inclusive_scan_over_group(g, x, binary_op, init);
}

/* JOINT_INCLUSIVE_SCAN'S OVERLOADS */

/// Marray<Complex> and Complex specialization
template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation, typename T,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              sycl::detail::is_pointer<InPtr>::value &&
              sycl::detail::is_pointer<OutPtr>::value &&
              (is_gencomplex_v<sycl::detail::remove_pointer_t<InPtr>> ||
               is_mgencomplex_v<sycl::detail::remove_pointer_t<
                   InPtr>>)&&(is_gencomplex_v<sycl::detail::
                                                  remove_pointer_t<OutPtr>> ||
                              is_mgencomplex_v<sycl::detail::remove_pointer_t<
                                  OutPtr>>)&&(is_gencomplex_v<T> ||
                                              is_mgencomplex_v<T>)&&cplx::
                  detail::is_binary_op_supported_v<BinaryOperation>>>
OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op, T init) {

  std::ptrdiff_t offset = g.get_local_linear_id();
  std::ptrdiff_t stride = g.get_local_linear_range();
  std::ptrdiff_t N = last - first;

  auto roundup = [=](const std::ptrdiff_t &v,
                     const std::ptrdiff_t &divisor) -> std::ptrdiff_t {
    return ((v + divisor - 1) / divisor) * divisor;
  };

  typename std::remove_const_t<typename sycl::detail::remove_pointer_t<InPtr>>
      x;
  typename sycl::detail::remove_pointer_t<OutPtr> carry = init;

  for (std::ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    std::ptrdiff_t i = chunk + offset;

    if (i < N)
      x = first[i];

    typename sycl::detail::remove_pointer_t<OutPtr> out =
        inclusive_scan_over_group(g, x, binary_op, carry);

    if (i < N)
      result[i] = out;

    carry = sycl::group_broadcast(g, out, stride - 1);
  }
  return result + N;
}

/// Marray<Complex> and Complex specialization
template <
    typename Group, typename InPtr, typename OutPtr, class BinaryOperation,
    typename = std::enable_if_t<
        sycl::is_group_v<std::decay_t<Group>> &&
        sycl::detail::is_pointer<InPtr>::value &&
        sycl::detail::is_pointer<OutPtr>::value &&
        (is_gencomplex_v<sycl::detail::remove_pointer_t<InPtr>> ||
         is_mgencomplex_v<sycl::detail::remove_pointer_t<
             InPtr>>)&&(is_gencomplex_v<sycl::detail::
                                            remove_pointer_t<OutPtr>> ||
                        is_mgencomplex_v<
                            sycl::detail::remove_pointer_t<OutPtr>>)&&cplx::
            detail::is_binary_op_supported_v<BinaryOperation>>>
OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op) {
  using T = typename sycl::detail::remove_pointer_t<InPtr>;

  auto init = cplx::detail::get_init<T, BinaryOperation>();

  return joint_inclusive_scan(g, first, last, result, binary_op, init);
}

/* EXCLUSIVE_SCAN_OVER_GROUP'S OVERLOADS */

/// Complex specialization
template <typename Group, typename V, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_genfloat_v<V> &&
              is_genfloat_v<T> &&
              cplx::detail::is_binary_op_supported_v<BinaryOperation>>>
complex<T> exclusive_scan_over_group(Group g, complex<V> x, complex<T> init,
                                     BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__

  complex<T> result;

  if constexpr (cplx::detail::is_plus_v<BinaryOperation>) {
    result.real(
        sycl::exclusive_scan_over_group(g, x.real(), init.real(), binary_op));
    result.imag(
        sycl::exclusive_scan_over_group(g, x.imag(), init.imag(), binary_op));
  } else {
    const auto flag = sycl::detail::spirv::group_scope<Group>::value;
    const auto operation =
        static_cast<unsigned int>(__spv::GroupOperation::ExclusiveScan);

    const auto spv_input =
        cplx::detail::construct_spv_complex(x.real(), x.imag());
    const auto spv_output = __spirv_GroupCMulINTEL(flag, operation, spv_input);

    result.real(spv_output.real);
    result.imag(spv_output.imag);
  }

  return result;
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

/// Marray<Complex> specialization
template <typename Group, typename V, std::size_t N, typename T, std::size_t S,
          class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> && is_gencomplex_v<V> &&
              is_gencomplex_v<T> &&
              cplx::detail::is_binary_op_supported_v<BinaryOperation>>>
sycl::marray<T, N> exclusive_scan_over_group(Group g, sycl::marray<V, N> x,
                                             sycl::marray<T, S> init,
                                             BinaryOperation binary_op) {
  sycl::marray<T, N> result;

  sycl::detail::loop<N>([&](size_t s) {
    result[s] = exclusive_scan_over_group(g, x[s], init[s], binary_op);
  });

  return result;
}

/// Marray<Complex> and Complex specialization
template <typename Group, typename T, class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              (is_gencomplex_v<T> || is_mgencomplex_v<T>)&&cplx::detail::
                  is_binary_op_supported_v<BinaryOperation>>>
T exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  auto init = cplx::detail::get_init<T, BinaryOperation>();

  return exclusive_scan_over_group(g, x, init, binary_op);
}

/* JOINT_EXCLUSIVE_SCAN'S OVERLOADS */

/// Marray<Complex> and Complex specialization
template <typename Group, typename InPtr, typename OutPtr, typename T,
          class BinaryOperation,
          typename = std::enable_if_t<
              sycl::is_group_v<std::decay_t<Group>> &&
              sycl::detail::is_pointer<InPtr>::value &&
              sycl::detail::is_pointer<OutPtr>::value &&
              (is_gencomplex_v<sycl::detail::remove_pointer_t<InPtr>> ||
               is_mgencomplex_v<sycl::detail::remove_pointer_t<
                   InPtr>>)&&(is_gencomplex_v<sycl::detail::
                                                  remove_pointer_t<OutPtr>> ||
                              is_mgencomplex_v<sycl::detail::remove_pointer_t<
                                  OutPtr>>)&&(is_gencomplex_v<T> ||
                                              is_mgencomplex_v<T>)&&cplx::
                  detail::is_binary_op_supported_v<BinaryOperation>>>
OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            T init, BinaryOperation binary_op) {
  std::ptrdiff_t offset = g.get_local_linear_id();
  std::ptrdiff_t stride = g.get_local_linear_range();
  std::ptrdiff_t N = last - first;

  auto roundup = [=](const std::ptrdiff_t &v,
                     const std::ptrdiff_t &divisor) -> std::ptrdiff_t {
    return ((v + divisor - 1) / divisor) * divisor;
  };

  typename std::remove_const_t<typename sycl::detail::remove_pointer_t<InPtr>>
      x;
  typename sycl::detail::remove_pointer_t<OutPtr> carry = init;

  for (std::ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    std::ptrdiff_t i = chunk + offset;
    if (i < N)
      x = first[i];

    typename sycl::detail::remove_pointer_t<OutPtr> out =
        exclusive_scan_over_group(g, x, carry, binary_op);

    if (i < N)
      result[i] = out;

    carry = sycl::group_broadcast(g, binary_op(out, x), stride - 1);
  }
  return result + N;
}

/// Marray<Complex> and Complex specialization
template <
    typename Group, typename InPtr, typename OutPtr, class BinaryOperation,
    typename = std::enable_if_t<
        sycl::is_group_v<std::decay_t<Group>> &&
        sycl::detail::is_pointer<InPtr>::value &&
        sycl::detail::is_pointer<OutPtr>::value &&
        (is_gencomplex_v<sycl::detail::remove_pointer_t<InPtr>> ||
         is_mgencomplex_v<sycl::detail::remove_pointer_t<
             InPtr>>)&&(is_gencomplex_v<sycl::detail::
                                            remove_pointer_t<OutPtr>> ||
                        is_mgencomplex_v<
                            sycl::detail::remove_pointer_t<OutPtr>>)&&cplx::
            detail::is_binary_op_supported_v<BinaryOperation>>>
OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op) {
  using T = typename sycl::detail::remove_pointer_t<InPtr>;

  auto init = cplx::detail::get_init<T, BinaryOperation>();

  return joint_exclusive_scan(g, first, last, result, init, binary_op);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

#undef _SYCL_EXT_CPLX_INLINE_VISIBILITY

#endif // SYCL_EXT_ONEAPI_COMPLEX
