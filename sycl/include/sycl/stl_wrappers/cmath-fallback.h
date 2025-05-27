#ifndef __CMATH_FALLBACK_H__
#define __CMATH_FALLBACK_H__

#ifdef __SYCL_DEVICE_ONLY__

#define __DPCPP_SYCL_DEVICE __attribute__((sycl_device_only, always_inline))

#define __DPCPP_SPIRV_MAP_UNARY(NAME, TYPE)                                    \
  __DPCPP_SYCL_DEVICE TYPE NAME(TYPE x) { return __spirv_ocl_##NAME(x); }

__DPCPP_SYCL_DEVICE long long abs(long long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE long abs(long n) { return n < 0 ? -n : n; }
__DPCPP_SYCL_DEVICE float abs(float x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE double abs(double x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE float fabs(float x) { return x < 0 ? -x : x; }
__DPCPP_SYCL_DEVICE double fabs(double x) { return x < 0 ? -x : x; }
__DPCPP_SPIRV_MAP_UNARY(acos, double);
__DPCPP_SPIRV_MAP_UNARY(acos, float);
__DPCPP_SPIRV_MAP_UNARY(acosh, double);
__DPCPP_SPIRV_MAP_UNARY(acosh, float);
__DPCPP_SPIRV_MAP_UNARY(asin, double);
__DPCPP_SPIRV_MAP_UNARY(asin, float);
__DPCPP_SPIRV_MAP_UNARY(asinh, double);
__DPCPP_SPIRV_MAP_UNARY(asinh, float);
__DPCPP_SYCL_DEVICE double scalbn(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE float scalbn(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE double scalbln(double x, long int exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
__DPCPP_SYCL_DEVICE float scalbln(float x, long int exp) {
  return __spirv_ocl_ldexp(x, (int)exp);
}
__DPCPP_SYCL_DEVICE double atan2(double x, double y) {
  return __spirv_ocl_atan2(x, y);
}
__DPCPP_SYCL_DEVICE float atan2(float x, float y) {
  return __spirv_ocl_atan2(x, y);
}
__DPCPP_SPIRV_MAP_UNARY(atan, double);
__DPCPP_SPIRV_MAP_UNARY(atan, float);
__DPCPP_SPIRV_MAP_UNARY(atanh, double);
__DPCPP_SPIRV_MAP_UNARY(atanh, float);
__DPCPP_SPIRV_MAP_UNARY(cbrt, double);
__DPCPP_SPIRV_MAP_UNARY(cbrt, float);
__DPCPP_SPIRV_MAP_UNARY(ceil, double);
__DPCPP_SPIRV_MAP_UNARY(ceil, float);
__DPCPP_SPIRV_MAP_UNARY(cos, double);
__DPCPP_SPIRV_MAP_UNARY(cos, float);
__DPCPP_SPIRV_MAP_UNARY(cosh, double);
__DPCPP_SPIRV_MAP_UNARY(cosh, float);
__DPCPP_SPIRV_MAP_UNARY(erfc, double);
__DPCPP_SPIRV_MAP_UNARY(erfc, float);
__DPCPP_SPIRV_MAP_UNARY(erf, double);
__DPCPP_SPIRV_MAP_UNARY(erf, float);
__DPCPP_SPIRV_MAP_UNARY(exp2, double);
__DPCPP_SPIRV_MAP_UNARY(exp2, float);
__DPCPP_SPIRV_MAP_UNARY(exp, double);
__DPCPP_SPIRV_MAP_UNARY(exp, float);
__DPCPP_SPIRV_MAP_UNARY(expm1, double);
__DPCPP_SPIRV_MAP_UNARY(expm1, float);
__DPCPP_SYCL_DEVICE double fdim(double x, double y) {
  return __spirv_ocl_fdim(x, y);
}
__DPCPP_SYCL_DEVICE float fdim(float x, float y) {
  return __spirv_ocl_fdim(x, y);
}
__DPCPP_SPIRV_MAP_UNARY(floor, double);
__DPCPP_SPIRV_MAP_UNARY(floor, float);
__DPCPP_SYCL_DEVICE double fma(double x, double y, double z) {
  return __spirv_ocl_fma(x, y, z);
}
__DPCPP_SYCL_DEVICE float fma(float x, float y, float z) {
  return __spirv_ocl_fma(x, y, z);
}
__DPCPP_SYCL_DEVICE double fmax(double x, double y) {
  return __spirv_ocl_fmax(x, y);
}
__DPCPP_SYCL_DEVICE float fmax(float x, float y) {
  return __spirv_ocl_fmax(x, y);
}
__DPCPP_SYCL_DEVICE double fmin(double x, double y) {
  return __spirv_ocl_fmin(x, y);
}
__DPCPP_SYCL_DEVICE float fmin(float x, float y) {
  return __spirv_ocl_fmin(x, y);
}
__DPCPP_SYCL_DEVICE double fmod(double x, double y) {
  return __spirv_ocl_fmod(x, y);
}
__DPCPP_SYCL_DEVICE float fmod(float x, float y) {
  return __spirv_ocl_fmod(x, y);
}
__DPCPP_SYCL_DEVICE double frexp(double x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
__DPCPP_SYCL_DEVICE float frexp(float x, int *exp) {
  return __spirv_ocl_frexp(x, exp);
}
__DPCPP_SYCL_DEVICE double hypot(double x, double y) {
  return __spirv_ocl_hypot(x, y);
}
__DPCPP_SYCL_DEVICE float hypot(float x, float y) {
  return __spirv_ocl_hypot(x, y);
}
__DPCPP_SYCL_DEVICE int ilogb(double x) { return __spirv_ocl_ilogb(x); }
__DPCPP_SYCL_DEVICE int ilogb(float x) { return __spirv_ocl_ilogb(x); }
__DPCPP_SYCL_DEVICE double ldexp(double x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SYCL_DEVICE float ldexp(float x, int exp) {
  return __spirv_ocl_ldexp(x, exp);
}
__DPCPP_SPIRV_MAP_UNARY(lgamma, double);
__DPCPP_SPIRV_MAP_UNARY(lgamma, float);
__DPCPP_SPIRV_MAP_UNARY(log10, double);
__DPCPP_SPIRV_MAP_UNARY(log10, float);
__DPCPP_SPIRV_MAP_UNARY(log1p, double);
__DPCPP_SPIRV_MAP_UNARY(log1p, float);
__DPCPP_SPIRV_MAP_UNARY(log2, double);
__DPCPP_SPIRV_MAP_UNARY(log2, float);
__DPCPP_SPIRV_MAP_UNARY(logb, double);
__DPCPP_SPIRV_MAP_UNARY(logb, float);
__DPCPP_SPIRV_MAP_UNARY(log, double);
__DPCPP_SPIRV_MAP_UNARY(log, float);
__DPCPP_SYCL_DEVICE double modf(double x, double *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
__DPCPP_SYCL_DEVICE float modf(float x, float *intpart) {
  return __spirv_ocl_modf(x, intpart);
}
__DPCPP_SYCL_DEVICE double nextafter(double x, double y) {
  return __spirv_ocl_nextafter(x, y);
}
__DPCPP_SYCL_DEVICE float nextafter(float x, float y) {
  return __spirv_ocl_nextafter(x, y);
}
__DPCPP_SYCL_DEVICE double pow(double x, double y) {
  return __spirv_ocl_pow(x, y);
}
__DPCPP_SYCL_DEVICE float pow(float x, float y) {
  return __spirv_ocl_pow(x, y);
}
__DPCPP_SYCL_DEVICE double remainder(double x, double y) {
  return __spirv_ocl_remainder(x, y);
}
__DPCPP_SYCL_DEVICE float remainder(float x, float y) {
  return __spirv_ocl_remainder(x, y);
}
__DPCPP_SYCL_DEVICE double remquo(double x, double y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}
__DPCPP_SYCL_DEVICE float remquo(float x, float y, int *q) {
  return __spirv_ocl_remquo(x, y, q);
}
__DPCPP_SPIRV_MAP_UNARY(rint, double);
__DPCPP_SPIRV_MAP_UNARY(rint, float);
__DPCPP_SPIRV_MAP_UNARY(round, double);
__DPCPP_SPIRV_MAP_UNARY(round, float);
__DPCPP_SPIRV_MAP_UNARY(sin, double);
__DPCPP_SPIRV_MAP_UNARY(sin, float);
__DPCPP_SPIRV_MAP_UNARY(sinh, double);
__DPCPP_SPIRV_MAP_UNARY(sinh, float);
__DPCPP_SPIRV_MAP_UNARY(sqrt, double);
__DPCPP_SPIRV_MAP_UNARY(sqrt, float);
__DPCPP_SPIRV_MAP_UNARY(tan, double);
__DPCPP_SPIRV_MAP_UNARY(tan, float);
__DPCPP_SPIRV_MAP_UNARY(tanh, double);
__DPCPP_SPIRV_MAP_UNARY(tanh, float);
__DPCPP_SPIRV_MAP_UNARY(tgamma, double);
__DPCPP_SPIRV_MAP_UNARY(tgamma, float);
__DPCPP_SPIRV_MAP_UNARY(trunc, double);
__DPCPP_SPIRV_MAP_UNARY(trunc, float);

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
using ::acosh;
using ::asin;
using ::asinh;
using ::atan;
using ::atan2;
using ::atanh;
using ::cbrt;
using ::ceil;
using ::div;
using ::ldiv;
using ::lldiv;
// using ::copysign;
using ::cos;
using ::cosh;
using ::erf;
using ::erfc;
using ::exp;
using ::exp2;
using ::expm1;
using ::fabs;
using ::fdim;
using ::floor;
using ::fma;
using ::fmax;
using ::fmin;
using ::fmod;
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
// using ::llabs;
// using ::llrint;
using ::log;
using ::log10;
using ::log1p;
using ::log2;
using ::logb;
// using ::lrint;
// using ::lround;
// using ::llround;
using ::modf;
// using ::nan;
// using ::nanf;
// using ::nearbyint;
using ::nextafter;
using ::pow;
using ::remainder;
using ::remquo;
using ::rint;
using ::round;
using ::scalbln;
using ::scalbn;
// using ::signbit;
using ::sin;
using ::sinh;
using ::sqrt;
using ::tan;
using ::tanh;
using ::tgamma;
using ::trunc;

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
