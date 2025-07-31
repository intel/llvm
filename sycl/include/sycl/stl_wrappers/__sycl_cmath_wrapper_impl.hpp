//==------------- __sycl_cmath_wrapper_impl.hpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_CMATH_WRAPPER_IMPL_HPP__
#define __SYCL_CMATH_WRAPPER_IMPL_HPP__

// This header defines device-side overloads of <cmath> functions based on
// their equivalent __spirv_ built-ins.

#ifdef __SYCL_DEVICE_ONLY__

// The 'sycl_device_only' attribute enables device-side overloading.
#define __SYCL_DEVICE __attribute__((sycl_device_only, always_inline))
#define __SYCL_DEVICE_C extern "C" __SYCL_DEVICE

#include <type_traits>

// Promotion templates: the C++ standard library provides overloads that allow
// arguments of math functions to be promoted. Any floating-point argument is
// allowed to accept any integer type, which should then be promoted to double.
// When multiple floating point arguments are available passing arguments with
// different precision should promote to the larger type. The template helpers
// below provide the machinery to define these promoting overloads.
template <typename T, bool = std::is_integral_v<T>> struct __sycl_promote {
  using type = double;
};

// Variant without ::type to allow SFINAE for non-promotable types.
template <typename T> struct __sycl_promote<T, false> {};

// float and double are left as is
template <> struct __sycl_promote<float> {
  using type = float;
};
template <> struct __sycl_promote<double> {
  using type = double;
};
// long double is not supported yet, so we don't define it,
// letting it SFINAE away too.
// We don't provide these overloads to makes sure that
// mixed precision calls that include long double are
// resolved to the "host" overload (defined by the real <cmath>),
// matching the behavior without promotion.
// Our long double overloads would fail to compile because
// we'd be trying to call SPIR-V built-ins that don't support long double.

// With two or three parameters we need to promote integers and possibly
// floating point types. We rely on operator+ with decltype to deduce the
// overall promotion type. This is only needed if at least one of the parameter
// is an integer, or if there's multiple different floating point types.
template <typename T, typename... Ts>
using __sycl_promote_t =
    std::enable_if_t<!std::conjunction_v<std::is_same<T, Ts>...> ||
                         std::is_integral_v<T> ||
                         (std::is_integral_v<Ts> || ...),
                     decltype((typename __sycl_promote<Ts>::type(0) + ... +
                               typename __sycl_promote<T>::type(0)))>;

// For each math built-in we need to define float and double overloads, an
// extern "C" float variant with the 'f' suffix, and a version that promotes
// integers or mixed precision floating-point parameters.
//
// TODO: Consider targets that don't have double support.
// TODO: Enable long double support where possible.
// TODO: float16_t and bfloat16_t support if the standard library
//       supports C++23.
// TODO: constexpr support for these functions (C++23, C++26)
//
// The following 4 macros provide an easy way to define these overloads for
// basic built-ins with one or two floating-point parameters.
//
// The double and the f suffixed versions must be defined in the global
// namespace, while the other overloads and templates should only be defined in
// the std namespace. Use __SYCL_SPIRV_MAP_UNARY_C, __SYCL_SPIRV_MAP_BINARY_C in
// the global namespace and __SYCL_SPIRV_MAP_UNARY_CXX,
// __SYCL_SPIRV_MAP_BINARY_CXX in the std namespace.
#define __SYCL_SPIRV_MAP_UNARY_C(NAME)                                         \
  __SYCL_DEVICE_C float NAME##f(float x) { return __spirv_ocl_##NAME(x); }     \
  __SYCL_DEVICE_C double NAME(double x) { return __spirv_ocl_##NAME(x); }

#define __SYCL_SPIRV_MAP_UNARY_CXX(NAME)                                       \
  using ::NAME;                                                                \
  using ::NAME##f;                                                             \
  __SYCL_DEVICE float NAME(float x) { return __spirv_ocl_##NAME(x); }          \
  template <typename T> __SYCL_DEVICE __sycl_promote_t<T> NAME(T x) {          \
    return __spirv_ocl_##NAME((double)x);                                      \
  }

#define __SYCL_SPIRV_MAP_BINARY_C(NAME)                                        \
  __SYCL_DEVICE_C float NAME##f(float x, float y) {                            \
    return __spirv_ocl_##NAME(x, y);                                           \
  }                                                                            \
  __SYCL_DEVICE_C double NAME(double x, double y) {                            \
    return __spirv_ocl_##NAME(x, y);                                           \
  }

#define __SYCL_SPIRV_MAP_BINARY_CXX(NAME)                                      \
  using ::NAME;                                                                \
  using ::NAME##f;                                                             \
  __SYCL_DEVICE float NAME(float x, float y) {                                 \
    return __spirv_ocl_##NAME(x, y);                                           \
  }                                                                            \
  template <typename T, typename U>                                            \
  __SYCL_DEVICE __sycl_promote_t<T, U> NAME(T x, U y) {                        \
    using type = __sycl_promote_t<T, U>;                                       \
    return __spirv_ocl_##NAME((type)x, (type)y);                               \
  }

/// Basic operations
//

__SYCL_DEVICE_C float fabsf(float x) { return x < 0 ? -x : x; }
__SYCL_DEVICE_C double fabs(double x) { return x < 0 ? -x : x; }

__SYCL_SPIRV_MAP_BINARY_C(fmod);
__SYCL_SPIRV_MAP_BINARY_C(remainder);

__SYCL_DEVICE_C float remquof(float x, float y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}
__SYCL_DEVICE_C double remquo(double x, double y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}

__SYCL_DEVICE_C float fmaf(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}
__SYCL_DEVICE_C double fma(double x, double y, double z) {
  return __spirv_ocl_fma(x, y, z);
}

__SYCL_SPIRV_MAP_BINARY_C(fmax);
__SYCL_SPIRV_MAP_BINARY_C(fmin);
__SYCL_SPIRV_MAP_BINARY_C(fdim);
// unsupported: nan

/// Exponential functions
//

__SYCL_SPIRV_MAP_UNARY_C(exp);
__SYCL_SPIRV_MAP_UNARY_C(exp2);
__SYCL_SPIRV_MAP_UNARY_C(expm1);
__SYCL_SPIRV_MAP_UNARY_C(log);
__SYCL_SPIRV_MAP_UNARY_C(log10);
__SYCL_SPIRV_MAP_UNARY_C(log2);
__SYCL_SPIRV_MAP_UNARY_C(log1p);

/// Power functions
//

__SYCL_SPIRV_MAP_BINARY_C(pow);
__SYCL_SPIRV_MAP_UNARY_C(sqrt);
__SYCL_SPIRV_MAP_UNARY_C(cbrt);
__SYCL_SPIRV_MAP_BINARY_C(hypot);

/// Trigonometric functions
//

__SYCL_SPIRV_MAP_UNARY_C(sin);
__SYCL_SPIRV_MAP_UNARY_C(cos);
__SYCL_SPIRV_MAP_UNARY_C(tan);
__SYCL_SPIRV_MAP_UNARY_C(asin);
__SYCL_SPIRV_MAP_UNARY_C(acos);
__SYCL_SPIRV_MAP_UNARY_C(atan);
__SYCL_SPIRV_MAP_BINARY_C(atan2);

/// Hyperbolic functions
//

__SYCL_SPIRV_MAP_UNARY_C(sinh);
__SYCL_SPIRV_MAP_UNARY_C(cosh);
__SYCL_SPIRV_MAP_UNARY_C(tanh);
__SYCL_SPIRV_MAP_UNARY_C(asinh);
__SYCL_SPIRV_MAP_UNARY_C(acosh);
__SYCL_SPIRV_MAP_UNARY_C(atanh);

/// Error and gamma functions
//

__SYCL_SPIRV_MAP_UNARY_C(erf);
__SYCL_SPIRV_MAP_UNARY_C(erfc);
__SYCL_SPIRV_MAP_UNARY_C(tgamma);
__SYCL_SPIRV_MAP_UNARY_C(lgamma);

/// Nearest integer floating-point operations
//

__SYCL_SPIRV_MAP_UNARY_C(ceil);
__SYCL_SPIRV_MAP_UNARY_C(floor);
__SYCL_SPIRV_MAP_UNARY_C(trunc);
__SYCL_SPIRV_MAP_UNARY_C(round);
// unsupported: lround, llround (no spirv mapping)
__SYCL_SPIRV_MAP_UNARY_C(rint);
// unsupported: lrint, llrint (no spirv mapping)

// unsupported (partially, no spirv mapping): nearbyint
#if defined(__NVPTX__)
extern "C" SYCL_EXTERNAL float __nv_nearbyintf(float);
extern "C" SYCL_EXTERNAL double __nv_nearbyint(double);
__SYCL_DEVICE_C float nearbyintf(float x) { return __nv_nearbyintf(x); }
__SYCL_DEVICE_C double nearbyint(double x) { return __nv_nearbyintf(x); }
#elif defined(__AMDGCN__)
extern "C" SYCL_EXTERNAL float __ocml_nearbyint_f32(float);
extern "C" SYCL_EXTERNAL double __ocml_nearbyint_f64(double);
__SYCL_DEVICE_C float nearbyintf(float x) { return __ocml_nearbyint_f32(x); }
__SYCL_DEVICE_C double nearbyint(double x) { return __ocml_nearbyint_f64(x); }
#endif

/// Floating-point manipulation functions
//

__SYCL_DEVICE_C float frexpf(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
__SYCL_DEVICE_C double frexp(double x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}

__SYCL_DEVICE_C float ldexpf(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__SYCL_DEVICE_C double ldexp(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

__SYCL_DEVICE_C float modff(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
__SYCL_DEVICE_C double modf(double x, double *intpart) {
  return __spirv_ocl_modf(x, intpart);
}

__SYCL_DEVICE_C float scalbnf(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__SYCL_DEVICE_C double scalbn(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

__SYCL_DEVICE_C float scalblnf(float x, long exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
__SYCL_DEVICE_C double scalbln(double x, long exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}

__SYCL_DEVICE_C int ilogbf(float x) { return __spirv_ocl_ilogb(x); }
__SYCL_DEVICE_C int ilogb(double x) { return __spirv_ocl_ilogb(x); }

__SYCL_SPIRV_MAP_UNARY_C(logb);
__SYCL_SPIRV_MAP_BINARY_C(nextafter);
// unsupported: nextforward
__SYCL_SPIRV_MAP_BINARY_C(copysign);

/// Classification and comparison

__SYCL_DEVICE_C int fpclassify(float x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, x);
}
__SYCL_DEVICE_C int fpclassify(double x) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL,
                              FP_ZERO, x);
}

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

// Basic operations
// using ::abs is already pulled in above
__SYCL_DEVICE float abs(float x) { return x < 0 ? -x : x; }
__SYCL_DEVICE double abs(double x) { return x < 0 ? -x : x; }

using ::fabs;
using ::fabsf;
__SYCL_DEVICE float fabs(float x) { return x < 0 ? -x : x; }
template <typename T> __SYCL_DEVICE __sycl_promote_t<T> fabs(T x) {
  return x < 0 ? -x : x;
}

using ::fdim;
using ::fdimf;
__SYCL_SPIRV_MAP_BINARY_CXX(fdim);

using ::fma;
using ::fmaf;
__SYCL_DEVICE float fma(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}
template <typename T, typename U, typename V>
__SYCL_DEVICE __sycl_promote_t<T, U, V> fma(T x, U y, V z) {
  using type = __sycl_promote_t<T, U, V>;
  return __spirv_ocl_fma((type)x, (type)y, (type)z);
}

__SYCL_SPIRV_MAP_BINARY_CXX(fmax);
__SYCL_SPIRV_MAP_BINARY_CXX(fmin);
__SYCL_SPIRV_MAP_BINARY_CXX(fmod);
__SYCL_SPIRV_MAP_BINARY_CXX(remainder);

using ::remquo;
using ::remquof;
__SYCL_DEVICE float remquo(float x, float y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}
template <typename T, typename U>
__SYCL_DEVICE __sycl_promote_t<T, U> remquo(T x, U y, int *q) {
  using type = __sycl_promote_t<T, U>;
  return __spirv_ocl_remquo((type)x, (type)y, q);
}
// using ::nan;
// using ::nanf;

// Exponential functions
__SYCL_SPIRV_MAP_UNARY_CXX(exp);
__SYCL_SPIRV_MAP_UNARY_CXX(exp2);
__SYCL_SPIRV_MAP_UNARY_CXX(expm1);
__SYCL_SPIRV_MAP_UNARY_CXX(log);
__SYCL_SPIRV_MAP_UNARY_CXX(log10);
__SYCL_SPIRV_MAP_UNARY_CXX(log1p);
__SYCL_SPIRV_MAP_UNARY_CXX(log2);

// Power functions
__SYCL_SPIRV_MAP_BINARY_CXX(pow);
__SYCL_SPIRV_MAP_UNARY_CXX(sqrt);
__SYCL_SPIRV_MAP_UNARY_CXX(cbrt);
__SYCL_SPIRV_MAP_BINARY_CXX(hypot);

// Trigonometric functions
__SYCL_SPIRV_MAP_UNARY_CXX(sin);
__SYCL_SPIRV_MAP_UNARY_CXX(cos);
__SYCL_SPIRV_MAP_UNARY_CXX(tan);
__SYCL_SPIRV_MAP_UNARY_CXX(asin);
__SYCL_SPIRV_MAP_UNARY_CXX(acos);
__SYCL_SPIRV_MAP_UNARY_CXX(atan);
__SYCL_SPIRV_MAP_BINARY_CXX(atan2);

// Hyperbloic functions
__SYCL_SPIRV_MAP_UNARY_CXX(sinh);
__SYCL_SPIRV_MAP_UNARY_CXX(cosh);
__SYCL_SPIRV_MAP_UNARY_CXX(tanh);
__SYCL_SPIRV_MAP_UNARY_CXX(asinh);
__SYCL_SPIRV_MAP_UNARY_CXX(acosh);
__SYCL_SPIRV_MAP_UNARY_CXX(atanh);

// Error and gamma functions
__SYCL_SPIRV_MAP_UNARY_CXX(erf);
__SYCL_SPIRV_MAP_UNARY_CXX(erfc);
__SYCL_SPIRV_MAP_UNARY_CXX(tgamma);
__SYCL_SPIRV_MAP_UNARY_CXX(lgamma);

// Nearest integer floating-point operations
__SYCL_SPIRV_MAP_UNARY_CXX(ceil);
__SYCL_SPIRV_MAP_UNARY_CXX(floor);
__SYCL_SPIRV_MAP_UNARY_CXX(trunc);
__SYCL_SPIRV_MAP_UNARY_CXX(round);
// using ::lround;
// using ::llround;
__SYCL_SPIRV_MAP_UNARY_CXX(rint);
// using ::lrint;
// using ::llrint;

#if defined(__NVPTX__) || defined(__AMDGCN__)
using ::nearbyint;
using ::nearbyintf;
#endif

#if defined(__NVPTX__)
__SYCL_DEVICE float nearbyint(float x) { return __nv_nearbyintf(x); }
#endif

#if defined(__AMDGCN__)
__SYCL_DEVICE float nearbyint(float x) { return __ocml_nearbyint_f32(x); }
#endif

// Floating-point manipulation functions
using ::frexp;
using ::frexpf;
__SYCL_DEVICE double frexp(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
template <typename T> __SYCL_DEVICE __sycl_promote_t<T> frexp(T x, int *exp) {
  return __spirv_ocl_frexp((double)x, exp);
}

using ::ilogb;
using ::ilogbf;
__SYCL_DEVICE int ilogb(float x) { return __spirv_ocl_ilogb(x); }
// ilogb needs a special template since its signature doesn't include the
// promoted type anywhere, so it needs to be specialized differently.
template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
__SYCL_DEVICE int ilogb(T x) {
  return __spirv_ocl_ilogb((double)x);
}

using ::ldexp;
using ::ldexpf;
__SYCL_DEVICE float ldexp(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
template <typename T> __SYCL_DEVICE __sycl_promote_t<T> ldexp(T x, int exp) {
  return __spirv_ocl_ldexp((double)x, exp);
}

__SYCL_SPIRV_MAP_UNARY_CXX(logb);

using ::modf;
using ::modff;
__SYCL_DEVICE float modf(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
// modf only supports integer x when the intpart is double.
template <typename T>
__SYCL_DEVICE __sycl_promote_t<T> modf(T x, double *intpart) {
  return __spirv_ocl_modf((double)x, intpart);
}

__SYCL_SPIRV_MAP_BINARY_CXX(nextafter);

using ::scalbln;
using ::scalblnf;
__SYCL_DEVICE float scalbln(float x, long exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
template <typename T> __SYCL_DEVICE __sycl_promote_t<T> scalbln(T x, long exp) {
  return __spirv_ocl_ldexp((double)x, (int)exp);
}

using ::scalbn;
using ::scalbnf;
__SYCL_DEVICE float scalbn(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
template <typename T> __SYCL_DEVICE __sycl_promote_t<T> scalbn(T x, int exp) {
  return __spirv_ocl_ldexp((double)x, exp);
}

// using ::nextforward
// using ::nextforwardf
__SYCL_SPIRV_MAP_BINARY_CXX(copysign);

// Classification and comparison
using ::fpclassify;
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

#if defined(_WIN32)
__SYCL_DEVICE_C double _Cosh(double x, double y) { return cosh(x) * y; }
__SYCL_DEVICE_C float _FCosh(float x, float y) { return coshf(x) * y; }
__SYCL_DEVICE_C short _Dtest(double *p) { return fpclassify(*p); }
__SYCL_DEVICE_C short _FDtest(float *p) { return fpclassify(*p); }
__SYCL_DEVICE_C double _Sinh(double x, double y) { return sinh(x) * y; }
__SYCL_DEVICE_C float _FSinh(float x, float y) { return sinhf(x) * y; }
__SYCL_DEVICE_C short _Exp(double *px, double y, short eoff) {
  return exp(*px) * ldexp(y, eoff);
}
__SYCL_DEVICE_C short _FExp(float *px, float y, short eoff) {
  return exp(*px) * ldexp(y, eoff);
}
__SYCL_DEVICE_C float _hypotf(float x, float y) { return hypotf(x, y); }
#endif // defined(_WIN32)

#undef __SYCL_SPIRV_MAP_BINARY_C
#undef __SYCL_SPIRV_MAP_BINARY_CXX
#undef __SYCL_SPIRV_MAP_UNARY_C
#undef __SYCL_SPIRV_MAP_UNARY_CXX
#undef __SYCL_DEVICE_C
#undef __SYCL_DEVICE
#endif // __SYCL_DEVICE_ONLY__
#endif // __SYCL_CMATH_WRAPPER_IMPL_HPP__
