#ifndef __CMATH_FALLBACK_H__
#define __CMATH_FALLBACK_H__

#ifdef __SYCL_DEVICE_ONLY__

#define __DPCPP_SYCL_DEVICE __attribute__((sycl_device_only, always_inline))
#define __DPCPP_SYCL_DEVICE_C                                                  \
  extern "C" __attribute__((sycl_device_only, always_inline))

#define __DPCPP_SPIRV_MAP_UNARY(NAME)                                          \
  __DPCPP_SYCL_DEVICE_C float NAME##f(float x) {                               \
    return __spirv_ocl_##NAME(x);                                              \
  }                                                                            \
  __DPCPP_SYCL_DEVICE float NAME(float x) { return __spirv_ocl_##NAME(x); }    \
  __DPCPP_SYCL_DEVICE double NAME(double x) { return __spirv_ocl_##NAME(x); }

#define __DPCPP_SPIRV_MAP_BINARY(NAME)                                         \
  __DPCPP_SYCL_DEVICE_C float NAME##f(float x, float y) {                      \
    return __spirv_ocl_##NAME(x, y);                                           \
  }                                                                            \
  __DPCPP_SYCL_DEVICE float NAME(float x, float y) {                           \
    return __spirv_ocl_##NAME(x, y);                                           \
  }                                                                            \
  __DPCPP_SYCL_DEVICE double NAME(double x, double y) {                        \
    return __spirv_ocl_##NAME(x, y);                                           \
  }

__DPCPP_SYCL_DEVICE long long abs(long long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE_C long long llabs(long long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE long abs(long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE int abs(int n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE_C long labs(long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE float abs(float x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE double abs(double x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE float fabs(float x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE_C float fabsf(float x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE double fabs(double x) { return x < 0 ? -x : x; }

__DPCPP_SPIRV_MAP_UNARY(acos);
__DPCPP_SPIRV_MAP_UNARY(acosh);
__DPCPP_SPIRV_MAP_UNARY(asin);
__DPCPP_SPIRV_MAP_UNARY(asinh);

__DPCPP_SYCL_DEVICE_C float scalbnf(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE float scalbn(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE double scalbn(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

__DPCPP_SYCL_DEVICE_C float scalblnf(float x, long int exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
__DPCPP_SYCL_DEVICE float scalbln(float x, long int exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
__DPCPP_SYCL_DEVICE double scalbln(double x, long int exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}

__DPCPP_SPIRV_MAP_BINARY(atan2);
__DPCPP_SPIRV_MAP_UNARY(atan);
__DPCPP_SPIRV_MAP_UNARY(atanh);
__DPCPP_SPIRV_MAP_UNARY(cbrt);
__DPCPP_SPIRV_MAP_UNARY(ceil);
__DPCPP_SPIRV_MAP_UNARY(cos);
__DPCPP_SPIRV_MAP_UNARY(cosh);
__DPCPP_SPIRV_MAP_UNARY(erfc);
__DPCPP_SPIRV_MAP_UNARY(erf);
__DPCPP_SPIRV_MAP_UNARY(exp2);
__DPCPP_SPIRV_MAP_UNARY(exp);
__DPCPP_SPIRV_MAP_UNARY(expm1);
__DPCPP_SPIRV_MAP_BINARY(fdim);
__DPCPP_SPIRV_MAP_UNARY(floor);

__DPCPP_SYCL_DEVICE_C float fmaf(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}
__DPCPP_SYCL_DEVICE float fma(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}
__DPCPP_SYCL_DEVICE double fma(double x, double y, double z) {
  return __spirv_ocl_fma(x, y, z);
}

__DPCPP_SPIRV_MAP_BINARY(fmax);
__DPCPP_SPIRV_MAP_BINARY(fmin);
__DPCPP_SPIRV_MAP_BINARY(fmod);

__DPCPP_SYCL_DEVICE_C float frexpf(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
__DPCPP_SYCL_DEVICE float frexp(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
__DPCPP_SYCL_DEVICE double frexp(double x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}

__DPCPP_SPIRV_MAP_BINARY(hypot);
__DPCPP_SYCL_DEVICE_C int ilogbf(float x) { return __spirv_ocl_ilogb(x); }
__DPCPP_SYCL_DEVICE int ilogb(float x) { return __spirv_ocl_ilogb(x); }
__DPCPP_SYCL_DEVICE int ilogb(double x) { return __spirv_ocl_ilogb(x); }

__DPCPP_SYCL_DEVICE_C float ldexpf(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE float ldexp(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE double ldexp(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}

__DPCPP_SPIRV_MAP_UNARY(lgamma);
__DPCPP_SPIRV_MAP_UNARY(log10);
__DPCPP_SPIRV_MAP_UNARY(log1p);
__DPCPP_SPIRV_MAP_UNARY(log2);
__DPCPP_SPIRV_MAP_UNARY(logb);
__DPCPP_SPIRV_MAP_UNARY(log);

__DPCPP_SYCL_DEVICE_C float modff(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
__DPCPP_SYCL_DEVICE float modf(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
__DPCPP_SYCL_DEVICE double modf(double x, double *intpart) {
  return __spirv_ocl_modf(x, intpart);
}

__DPCPP_SPIRV_MAP_BINARY(nextafter);
__DPCPP_SPIRV_MAP_BINARY(pow);
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
__DPCPP_SPIRV_MAP_UNARY(rint);
__DPCPP_SPIRV_MAP_UNARY(round);
__DPCPP_SPIRV_MAP_UNARY(sin);
__DPCPP_SPIRV_MAP_UNARY(sinh);
__DPCPP_SPIRV_MAP_UNARY(sqrt);
__DPCPP_SPIRV_MAP_UNARY(tan);
__DPCPP_SPIRV_MAP_UNARY(tanh);
__DPCPP_SPIRV_MAP_UNARY(tgamma);
__DPCPP_SPIRV_MAP_UNARY(trunc);

__DPCPP_SYCL_DEVICE div_t div(int x, int y) { return {x / y, x % y}; }

__DPCPP_SYCL_DEVICE ldiv_t ldiv(long x, long y) { return {x / y, x % y}; }

__DPCPP_SYCL_DEVICE lldiv_t ldiv(long long x, long long y) {
  return {x / y, x % y};
}

#ifdef _LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_NAMESPACE_STD
#else
namespace std {
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_BEGIN_NAMESPACE_VERSION
#endif
#endif

using ::abs;
using ::acos;
using ::acosf;
using ::acosh;
using ::acoshf;
using ::asin;
using ::asinf;
using ::asinh;
using ::asinhf;
using ::atan;
using ::atan2;
using ::atan2f;
using ::atanf;
using ::atanh;
using ::atanhf;
using ::cbrt;
using ::cbrtf;
using ::ceil;
using ::ceilf;
using ::div;
using ::labs;
using ::ldiv;
using ::llabs;
using ::lldiv;
// using ::copysign;
using ::cos;
using ::cosf;
using ::cosh;
using ::coshf;
using ::erf;
using ::erfc;
using ::erfcf;
using ::erff;
using ::exp;
using ::exp2;
using ::exp2f;
using ::expf;
using ::expm1;
using ::expm1f;
using ::fabs;
using ::fabsf;
using ::fdim;
using ::fdimf;
using ::floor;
using ::floorf;
using ::fmaf;
using ::fmaxf;
using ::fminf;
using ::fmod;
using ::fmodf;
// using ::fpclassify;
using ::frexp;
using ::hypot;
using ::ilogb;
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
// using ::labs;
using ::ldexp;
using ::lgamma;
using ::lgammaf;
// using ::llabs;
// using ::llrint;
using ::log;
using ::log10;
using ::log10f;
using ::log1p;
using ::log1pf;
using ::log2;
using ::log2f;
using ::logb;
using ::logbf;
using ::logf;
// using ::lrint;
// using ::lround;
// using ::llround;
using ::modf;
using ::modff;
// using ::nan;
// using ::nanf;
// using ::nearbyint;
using ::nextafter;
using ::nextafterf;
using ::pow;
using ::powf;
using ::remainder;
using ::remainderf;
using ::remquo;
using ::remquof;
using ::rint;
using ::rintf;
using ::round;
using ::roundf;
using ::scalbln;
using ::scalblnf;
using ::scalbn;
using ::scalbnf;
// using ::signbit;
using ::sin;
using ::sinf;
using ::sinh;
using ::sinhf;
using ::sqrt;
using ::sqrtf;
using ::tan;
using ::tanf;
using ::tanh;
using ::tanhf;
using ::tgamma;
using ::tgammaf;
using ::trunc;
using ::truncf;

#ifdef _LIBCPP_END_NAMESPACE_STD
_LIBCPP_END_NAMESPACE_STD
#else
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_END_NAMESPACE_VERSION
#endif
} // namespace std
#endif

#undef __DPCPP_SPIRV_MAP_UNARY
#undef __DPCPP_SYCL_DEVICE
#endif
#endif
