//==------------- cmath-fallback.h -----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CMATH_FALLBACK_H__
#define __CMATH_FALLBACK_H__

// This header defines device-side overloads of <cmath> functions based on
// their equivalent __spirv_ built-ins.

#ifdef __SYCL_DEVICE_ONLY__

// The 'sycl_device_only' attribute enables device-side overloading.
#define __DPCPP_SYCL_DEVICE __attribute__((sycl_device_only, always_inline))
#define __DPCPP_SYCL_DEVICE_C                                                  \
  extern "C" __attribute__((sycl_device_only, always_inline))

// Promotion templates: the C++ standard library provides overloads that allow
// arguments of math functions to be promoted. Any floating-point argument is
// allowed to accept any integer type, which should then be promoted to double.
// When multiple floating point arguments are available passing arguments with
// different precision should promote to the larger type. The template helpers
// below provide the machinery to define these promoting overloads.
template <typename T, bool = (std::is_integral<T>::value ||
                              std::is_floating_point<T>::value)>
struct __dpcpp_promote {
private:
  // Integer types are promoted to double.
  template <typename U>
  static typename std::enable_if<std::is_integral<U>::value, double>::type
  test();

  // Floating point types are used as-is.
  template <typename U>
  static typename std::enable_if<std::is_floating_point<U>::value, U>::type
  test();

public:
  // We rely on dummy templated methods and decltype to select the right type
  // based on the input T.
  typedef decltype(test<T>()) type;
};

// Variant without ::type to allow SFINAE for non promotable types.
template <typename T> struct __dpcpp_promote<T, false> {};

// With a single paramter we only need to promote integers.
template <typename T>
using __dpcpp_promote_1 = std::enable_if<std::is_integral<T>::value, double>;

// With two or three parameters we need to promote integers and possibly
// floating point types. We rely on operator+ with decltype to deduce the
// overall promotion type. This is only needed if at least one of the parameter
// is an integer, or if there's multiple different floating point types.
template <typename T, typename U>
using __dpcpp_promote_2 =
    std::enable_if<!std::is_same<T, U>::value || std::is_integral<T>::value ||
                       std::is_integral<U>::value,
                   decltype(typename __dpcpp_promote<T>::type(0) +
                            typename __dpcpp_promote<U>::type(0))>;

template <typename T, typename U, typename V>
using __dpcpp_promote_3 =
    std::enable_if<!(std::is_same<T, U>::value && std::is_same<U, V>::value) ||
                       std::is_integral<T>::value ||
                       std::is_integral<U>::value || std::is_integral<V>::value,
                   decltype(typename __dpcpp_promote<T>::type(0) +
                            typename __dpcpp_promote<U>::type(0) +
                            typename __dpcpp_promote<V>::type(0))>;

// For each math built-in we need to define float and double overloads, an
// extern "C" float variant with the 'f' suffix, and a version that promotes to
// double if any floating-point parameter passed is an integer.
//
// TODO: Consider targets that don't have double support
// TODO: Enable long double support where possible
//
// The following two macros provide an easy way to define these overloads for
// basic built-ins with one or two floating-point parameters.
#define __DPCPP_SPIRV_MAP_UNARY(NAME)                                          \
  __DPCPP_SYCL_DEVICE_C float NAME##f(float x) {                               \
    return __spirv_ocl_##NAME(x);                                              \
  }                                                                            \
  __DPCPP_SYCL_DEVICE float NAME(float x) { return __spirv_ocl_##NAME(x); }    \
  __DPCPP_SYCL_DEVICE double NAME(double x) { return __spirv_ocl_##NAME(x); }  \
  template <typename T>                                                        \
  __DPCPP_SYCL_DEVICE typename __dpcpp_promote_1<T>::type NAME(T x) {          \
    return __spirv_ocl_##NAME((double)x);                                      \
  }

#define __DPCPP_SPIRV_MAP_BINARY(NAME)                                         \
  __DPCPP_SYCL_DEVICE_C float NAME##f(float x, float y) {                      \
    return __spirv_ocl_##NAME(x, y);                                           \
  }                                                                            \
  __DPCPP_SYCL_DEVICE float NAME(float x, float y) {                           \
    return __spirv_ocl_##NAME(x, y);                                           \
  }                                                                            \
  __DPCPP_SYCL_DEVICE double NAME(double x, double y) {                        \
    return __spirv_ocl_##NAME(x, y);                                           \
  }                                                                            \
  template <typename T, typename U>                                            \
  __DPCPP_SYCL_DEVICE __dpcpp_promote_2<T, U>::type NAME(T x, U y) {           \
    typedef typename __dpcpp_promote_2<T, U>::type type;                       \
    return __spirv_ocl_##NAME((type)x, (type)y);                               \
  }

/// <cstdlib>
// FIXME: Move this to a cstdlib fallback header

__DPCPP_SYCL_DEVICE div_t div(int x, int y) { return {x / y, x % y}; }
__DPCPP_SYCL_DEVICE ldiv_t ldiv(long x, long y) { return {x / y, x % y}; }
__DPCPP_SYCL_DEVICE lldiv_t ldiv(long long x, long long y) {
  return {x / y, x % y};
}

__DPCPP_SYCL_DEVICE long long abs(long long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE_C long long llabs(long long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE long abs(long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE int abs(int n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE_C long labs(long n) { return n < 0 ? -n : n; }

/// Basic operations
//

__DPCPP_SYCL_DEVICE float abs(float x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE double abs(double x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE float fabs(float x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE_C float fabsf(float x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE double fabs(double x) { return x < 0 ? -x : x; }
template <typename T>
__DPCPP_SYCL_DEVICE typename __dpcpp_promote_1<T>::type fabs(T x) {
  return x < 0 ? -x : x;
}

__DPCPP_SPIRV_MAP_BINARY(fmod);
__DPCPP_SPIRV_MAP_BINARY(remainder);

__DPCPP_SYCL_DEVICE_C float remquof(float x, float y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}
__DPCPP_SYCL_DEVICE float remquo(float x, float y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}
__DPCPP_SYCL_DEVICE double remquo(double x, double y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}
template <typename T, typename U>
__DPCPP_SYCL_DEVICE typename __dpcpp_promote_2<T, U>::type remquo(T x, U y,
                                                                  int *q) {
  typedef typename __dpcpp_promote_2<T, U>::type type;
  return __spirv_ocl_remquo((type)x, (type)y, q);
}

__DPCPP_SYCL_DEVICE_C float fmaf(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}
__DPCPP_SYCL_DEVICE float fma(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}
__DPCPP_SYCL_DEVICE double fma(double x, double y, double z) {
  return __spirv_ocl_fma(x, y, z);
}
template <typename T, typename U, typename V>
__DPCPP_SYCL_DEVICE typename __dpcpp_promote_3<T, U, V>::type fma(T x, U y,
                                                                  V z) {
  typedef typename __dpcpp_promote_3<T, U, V>::type type;
  return __spirv_ocl_fma((type)x, (type)y, (type)z);
}

__DPCPP_SPIRV_MAP_BINARY(fmax);
__DPCPP_SPIRV_MAP_BINARY(fmin);
__DPCPP_SPIRV_MAP_BINARY(fdim);
// unsupported: nan

/// Exponential functions
//

__DPCPP_SPIRV_MAP_UNARY(exp);
__DPCPP_SPIRV_MAP_UNARY(exp2);
__DPCPP_SPIRV_MAP_UNARY(expm1);
__DPCPP_SPIRV_MAP_UNARY(log);
__DPCPP_SPIRV_MAP_UNARY(log10);
__DPCPP_SPIRV_MAP_UNARY(log2);
__DPCPP_SPIRV_MAP_UNARY(log1p);

/// Power functions
//

__DPCPP_SPIRV_MAP_BINARY(pow);
__DPCPP_SPIRV_MAP_UNARY(sqrt);
__DPCPP_SPIRV_MAP_UNARY(cbrt);
__DPCPP_SPIRV_MAP_BINARY(hypot);

/// Trigonometric functions
//

__DPCPP_SPIRV_MAP_UNARY(sin);
__DPCPP_SPIRV_MAP_UNARY(cos);
__DPCPP_SPIRV_MAP_UNARY(tan);
__DPCPP_SPIRV_MAP_UNARY(asin);
__DPCPP_SPIRV_MAP_UNARY(acos);
__DPCPP_SPIRV_MAP_UNARY(atan);
__DPCPP_SPIRV_MAP_BINARY(atan2);

/// Hyperbolic functions
//

__DPCPP_SPIRV_MAP_UNARY(sinh);
__DPCPP_SPIRV_MAP_UNARY(cosh);
__DPCPP_SPIRV_MAP_UNARY(tanh);
__DPCPP_SPIRV_MAP_UNARY(asinh);
__DPCPP_SPIRV_MAP_UNARY(acosh);
__DPCPP_SPIRV_MAP_UNARY(atanh);

/// Error and gamma functions
//

__DPCPP_SPIRV_MAP_UNARY(erf);
__DPCPP_SPIRV_MAP_UNARY(erfc);
__DPCPP_SPIRV_MAP_UNARY(tgamma);
__DPCPP_SPIRV_MAP_UNARY(lgamma);

/// Nearest integer floating-point operations
//

__DPCPP_SPIRV_MAP_UNARY(ceil);
__DPCPP_SPIRV_MAP_UNARY(floor);
__DPCPP_SPIRV_MAP_UNARY(trunc);
__DPCPP_SPIRV_MAP_UNARY(round);
// unsupported: lround, llround (no spirv mapping)
__DPCPP_SPIRV_MAP_UNARY(rint);
// unsupported: lrint, llrint (no spirv mapping)

// unsupported (partially, no spirv mapping): nearbyint
#if defined(__NVPTX__)
extern "C" SYCL_EXTERNAL float __nv_nearbyintf(float);
extern "C" SYCL_EXTERNAL double __nv_nearbyint(double);
__DPCPP_SYCL_DEVICE_C float nearbyintf(float x) { return __nv_nearbyintf(x); }
__DPCPP_SYCL_DEVICE float nearbyint(float x) { return __nv_nearbyintf(x); }
__DPCPP_SYCL_DEVICE double nearbyint(double x) { return __nv_nearbyintf(x); }
#elif defined(__AMDGCN__)
extern "C" SYCL_EXTERNAL float __ocml_nearbyint_f32(float);
extern "C" SYCL_EXTERNAL double __ocml_nearbyint_f64(double);
__DPCPP_SYCL_DEVICE_C float nearbyintf(float x) {
  return __ocml_nearbyint_f32(x);
}
__DPCPP_SYCL_DEVICE float nearbyint(float x) { return __ocml_nearbyint_f32(x); }
__DPCPP_SYCL_DEVICE double nearbyint(double x) {
  return __ocml_nearbyint_f64(x);
}
#endif

/// Floating-point manipulation functions
//

__DPCPP_SYCL_DEVICE_C float frexpf(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
__DPCPP_SYCL_DEVICE float frexp(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
__DPCPP_SYCL_DEVICE double frexp(double x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
template <typename T>
__DPCPP_SYCL_DEVICE typename __dpcpp_promote_1<T>::type frexp(T x, int *exp) {
  return __spirv_ocl_frexp((double)x, exp);
}

__DPCPP_SYCL_DEVICE_C float ldexpf(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE float ldexp(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE double ldexp(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
template <typename T>
__DPCPP_SYCL_DEVICE typename __dpcpp_promote_1<T>::type ldexp(T x, int exp) {
  return __spirv_ocl_ldexp((double)x, exp);
}

__DPCPP_SYCL_DEVICE_C float modff(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
__DPCPP_SYCL_DEVICE float modf(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
__DPCPP_SYCL_DEVICE double modf(double x, double *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
// modf only supports integer x when the intpart is double
template <typename T>
__DPCPP_SYCL_DEVICE typename __dpcpp_promote_1<T>::type modf(T x,
                                                             double *intpart) {
  return __spirv_ocl_modf((double)x, intpart);
}

__DPCPP_SYCL_DEVICE_C float scalbnf(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE float scalbn(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE double scalbn(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
template <typename T>
__DPCPP_SYCL_DEVICE typename __dpcpp_promote_1<T>::type scalbn(T x, int exp) {
  return __spirv_ocl_ldexp((double)x, exp);
}

__DPCPP_SYCL_DEVICE_C float scalblnf(float x, long exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
__DPCPP_SYCL_DEVICE float scalbln(float x, long exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
__DPCPP_SYCL_DEVICE double scalbln(double x, long exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
template <typename T>
__DPCPP_SYCL_DEVICE typename __dpcpp_promote_1<T>::type scalbln(T x, long exp) {
  return __spirv_ocl_ldexp((double)x, (int)exp);
}

__DPCPP_SYCL_DEVICE_C int ilogbf(float x) { return __spirv_ocl_ilogb(x); }
__DPCPP_SYCL_DEVICE int ilogb(float x) { return __spirv_ocl_ilogb(x); }
__DPCPP_SYCL_DEVICE int ilogb(double x) { return __spirv_ocl_ilogb(x); }
template <typename T, typename std::enable_if<std::is_integral<T>::value,
                                              bool>::type = true>
__DPCPP_SYCL_DEVICE double ilogb(T x) {
  return __spirv_ocl_ilogb((double)x);
}

__DPCPP_SPIRV_MAP_UNARY(logb);
__DPCPP_SPIRV_MAP_BINARY(nextafter);
// unsupported: nextforward
__DPCPP_SPIRV_MAP_BINARY(copysign);

/// Classification and comparison
//

// unsupported: fpclassify
// unsupported: isfinite
// unsupported: isinf
// unsupported: isnan
// unsupported: isnormal
// unsupported: signbit
// unsupported: isgreater
// unsupported: isgreaterequal
// unsupported: isless
// unsupported: islessequal
// unsupported: islessgreated
// unsupported: isunordered

// Now drag all of the overloads we've just defined in the std namespace. For
// the overloads to work properly we need to ensure our namespace matches
// exactly the one of the system C++ library.
#ifdef _LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_NAMESPACE_STD
#else
namespace std {
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_BEGIN_NAMESPACE_VERSION
#endif
#endif

// <cstdlib>
using ::div;
using ::labs;
using ::ldiv;
using ::llabs;
using ::lldiv;

// Basic operations
using ::abs;
using ::fabs;
using ::fabsf;
using ::fdim;
using ::fdimf;
using ::fma;
using ::fmaf;
using ::fmax;
using ::fmaxf;
using ::fmin;
using ::fminf;
using ::fmod;
using ::fmodf;
using ::remainder;
using ::remainderf;
using ::remquo;
using ::remquof;
// using ::nan;
// using ::nanf;

// Exponential functions
using ::exp;
using ::exp2;
using ::exp2f;
using ::expf;
using ::expm1;
using ::expm1f;
using ::log;
using ::log10;
using ::log10f;
using ::log1p;
using ::log1pf;
using ::log2;
using ::log2f;
using ::logf;

// Power functions
using ::cbrt;
using ::cbrtf;
using ::hypot;
using ::hypotf;
using ::pow;
using ::powf;
using ::sqrt;
using ::sqrtf;

// Trigonometric functions
using ::acos;
using ::acosf;
using ::asin;
using ::asinf;
using ::atan;
using ::atan2;
using ::atan2f;
using ::atanf;
using ::cos;
using ::cosf;
using ::sin;
using ::sinf;
using ::tan;
using ::tanf;

// Hyperbloic functions
using ::acosh;
using ::acoshf;
using ::asinh;
using ::asinhf;
using ::atanh;
using ::atanhf;
using ::cosh;
using ::coshf;
using ::sinh;
using ::sinhf;
using ::tanh;
using ::tanhf;

// Error and gamma functions
using ::erf;
using ::erfc;
using ::erfcf;
using ::erff;
using ::lgamma;
using ::lgammaf;
using ::tgamma;
using ::tgammaf;

// Nearest integer floating-point operations
using ::ceil;
using ::ceilf;
using ::floor;
using ::floorf;
using ::round;
using ::roundf;
using ::trunc;
using ::truncf;
// using ::lround;
// using ::llround;
using ::rint;
using ::rintf;
// using ::lrint;
// using ::llrint;

#if defined(__NVPTX__) || defined(__AMDGCN__)
using ::nearbyint;
using ::nearbyintf;
#endif

// Floating-point manipulation functions
using ::frexp;
using ::frexpf;
using ::ilogb;
using ::ilogbf;
using ::ldexp;
using ::ldexpf;
using ::logb;
using ::logbf;
using ::modf;
using ::modff;
using ::nextafter;
using ::nextafterf;
using ::scalbln;
using ::scalblnf;
using ::scalbn;
using ::scalbnf;
// using ::nextforward
// using ::nextforwardf
using ::copysign;
using ::copysignf;

// Classification and comparison
// using ::fpclassify;
// using ::isfinite;
// using ::isgreater;
// using ::isgreaterequal;
// using ::isinf;
// using ::isless;
// using ::islessequal;
// using ::islessgreater;
// using ::isnan;
// using ::isnormal;
// using ::isunordered;
// using ::signbit;

#ifdef _LIBCPP_END_NAMESPACE_STD
_LIBCPP_END_NAMESPACE_STD
#else
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_END_NAMESPACE_VERSION
#endif
} // namespace std
#endif

#undef __DPCPP_SPIRV_MAP_BINARY
#undef __DPCPP_SPIRV_MAP_UNARY
#undef __DPCPP_SYCL_DEVICE_C
#undef __DPCPP_SYCL_DEVICE
#endif
#endif
