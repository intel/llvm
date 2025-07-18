// REQUIRES: cuda
// Note: This isn't really target specific and should be switched to spir when
// it's enabled for it.

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -S -Xclang -emit-llvm -fsycl-device-only %s -o - | FileCheck %s

#include <cmath>

// CHECK-LABEL: entry
__attribute__((sycl_device)) void entry(float *fp, double *dp, int *ip,
                                        long *lp, long long *llp, float *rf,
                                        double *rd, int *ri) {
  // Use an incrementing index to prevent the compiler from optimizing some
  // calls that would store to the same address.
  int idx = 0;

  // For each supported standard math built-in, we test that the following
  // overloads are properly mapped to __spirv_ built-ins:
  //
  // * Float only.
  // * Float only with 'f' suffix.
  // * Double only.
  // * Integer promotion.
  // * Mixed floating point promotion (when applicable).
  //
  // CHECK: __spirv_ocl_fmodff
  rf[idx++] = std::fmod(fp[0], fp[1]);
  // CHECK: __spirv_ocl_fmodff
  rf[idx++] = std::fmodf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_fmoddd
  rd[idx++] = std::fmod(dp[0], dp[1]);
  // CHECK: __spirv_ocl_fmoddd
  rd[idx++] = std::fmod(fp[0], ip[1]);
  // CHECK: __spirv_ocl_fmoddd
  rd[idx++] = std::fmod(fp[0], dp[1]);

  // CHECK: __spirv_ocl_remainderff
  rf[idx++] = std::remainder(fp[0], fp[1]);
  // CHECK: __spirv_ocl_remainderff
  rf[idx++] = std::remainderf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_remainderdd
  rd[idx++] = std::remainder(dp[0], dp[1]);
  // CHECK: __spirv_ocl_remainderdd
  rd[idx++] = std::remainder(fp[0], ip[1]);
  // CHECK: __spirv_ocl_remainderdd
  rd[idx++] = std::remainder(fp[0], dp[1]);

  // CHECK: __spirv_ocl_remquoff
  rf[idx++] = std::remquo(fp[0], fp[1], ip);
  // CHECK: __spirv_ocl_remquoff
  rf[idx++] = std::remquof(fp[2], fp[1], ip);
  // CHECK: __spirv_ocl_remquodd
  rd[idx++] = std::remquo(dp[0], dp[1], ip);
  // CHECK: __spirv_ocl_remquodd
  rd[idx++] = std::remquo(fp[0], ip[1], ip);
  // CHECK: __spirv_ocl_remquodd
  rd[idx++] = std::remquo(fp[0], dp[1], ip);

  // CHECK: __spirv_ocl_fmaff
  rf[idx++] = std::fma(fp[0], fp[1], fp[2]);
  // CHECK: __spirv_ocl_fmaff
  rf[idx++] = std::fmaf(fp[3], fp[1], fp[2]);
  // CHECK: __spirv_ocl_fmadd
  rd[idx++] = std::fma(dp[0], dp[1], dp[2]);
  // CHECK: __spirv_ocl_fmadd
  rd[idx++] = std::fma(fp[0], ip[1], fp[2]);
  // CHECK: __spirv_ocl_fmadd
  rd[idx++] = std::fma(fp[0], dp[1], fp[2]);

  // CHECK: __spirv_ocl_fmaxff
  rf[idx++] = std::fmax(fp[0], fp[1]);
  // CHECK: __spirv_ocl_fmaxff
  rf[idx++] = std::fmaxf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_fmaxdd
  rd[idx++] = std::fmax(dp[0], dp[1]);
  // CHECK: __spirv_ocl_fmaxdd
  rd[idx++] = std::fmax(fp[0], ip[1]);
  // CHECK: __spirv_ocl_fmaxdd
  rd[idx++] = std::fmax(fp[0], dp[1]);

  // CHECK: __spirv_ocl_fminff
  rf[idx++] = std::fmin(fp[0], fp[1]);
  // CHECK: __spirv_ocl_fminff
  rf[idx++] = std::fminf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_fmindd
  rd[idx++] = std::fmin(dp[0], dp[1]);
  // CHECK: __spirv_ocl_fmindd
  rd[idx++] = std::fmin(fp[0], ip[1]);
  // CHECK: __spirv_ocl_fmindd
  rd[idx++] = std::fmin(fp[0], dp[1]);

  // CHECK: __spirv_ocl_fdimff
  rf[idx++] = std::fdim(fp[0], fp[1]);
  // CHECK: __spirv_ocl_fdimff
  rf[idx++] = std::fdimf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_fdimdd
  rd[idx++] = std::fdim(dp[0], dp[1]);
  // CHECK: __spirv_ocl_fdimdd
  rd[idx++] = std::fdim(fp[0], ip[1]);
  // CHECK: __spirv_ocl_fdimdd
  rd[idx++] = std::fdim(fp[0], dp[1]);

  // CHECK: __spirv_ocl_expf
  rf[idx++] = std::exp(fp[0]);
  // CHECK: __spirv_ocl_expf
  rf[idx++] = std::expf(fp[1]);
  // CHECK: __spirv_ocl_expd
  rd[idx++] = std::exp(dp[0]);
  // CHECK: __spirv_ocl_expd
  rd[idx++] = std::exp(ip[0]);

  // CHECK: __spirv_ocl_exp2f
  rf[idx++] = std::exp2(fp[0]);
  // CHECK: __spirv_ocl_exp2f
  rf[idx++] = std::exp2f(fp[1]);
  // CHECK: __spirv_ocl_exp2d
  rd[idx++] = std::exp2(dp[0]);
  // CHECK: __spirv_ocl_exp2d
  rd[idx++] = std::exp2(ip[0]);

  // CHECK: __spirv_ocl_expm1f
  rf[idx++] = std::expm1(fp[0]);
  // CHECK: __spirv_ocl_expm1f
  rf[idx++] = std::expm1f(fp[1]);
  // CHECK: __spirv_ocl_expm1d
  rd[idx++] = std::expm1(dp[0]);
  // CHECK: __spirv_ocl_expm1d
  rd[idx++] = std::expm1(ip[0]);

  // CHECK: __spirv_ocl_logf
  rf[idx++] = std::log(fp[0]);
  // CHECK: __spirv_ocl_logf
  rf[idx++] = std::logf(fp[1]);
  // CHECK: __spirv_ocl_logd
  rd[idx++] = std::log(dp[0]);
  // CHECK: __spirv_ocl_logd
  rd[idx++] = std::log(ip[0]);

  // CHECK: __spirv_ocl_log10f
  rf[idx++] = std::log10(fp[0]);
  // CHECK: __spirv_ocl_log10f
  rf[idx++] = std::log10f(fp[1]);
  // CHECK: __spirv_ocl_log10d
  rd[idx++] = std::log10(dp[0]);
  // CHECK: __spirv_ocl_log10d
  rd[idx++] = std::log10(ip[0]);

  // CHECK: __spirv_ocl_log2f
  rf[idx++] = std::log2(fp[0]);
  // CHECK: __spirv_ocl_log2f
  rf[idx++] = std::log2f(fp[1]);
  // CHECK: __spirv_ocl_log2d
  rd[idx++] = std::log2(dp[0]);
  // CHECK: __spirv_ocl_log2d
  rd[idx++] = std::log2(ip[0]);

  // CHECK: __spirv_ocl_log1pf
  rf[idx++] = std::log1p(fp[0]);
  // CHECK: __spirv_ocl_log1pf
  rf[idx++] = std::log1pf(fp[1]);
  // CHECK: __spirv_ocl_log1pd
  rd[idx++] = std::log1p(dp[0]);
  // CHECK: __spirv_ocl_log1pd
  rd[idx++] = std::log1p(ip[0]);

  // CHECK: __spirv_ocl_powff
  rf[idx++] = std::pow(fp[0], fp[1]);
  // CHECK: __spirv_ocl_powff
  rf[idx++] = std::powf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_powdd
  rd[idx++] = std::pow(dp[0], dp[1]);
  // CHECK: __spirv_ocl_powdd
  rd[idx++] = std::pow(ip[0], fp[1]);
  // CHECK: __spirv_ocl_powdd
  rd[idx++] = std::pow(dp[0], fp[1]);

  // CHECK: __spirv_ocl_sqrtf
  rf[idx++] = std::sqrt(fp[0]);
  // CHECK: __spirv_ocl_sqrtf
  rf[idx++] = std::sqrtf(fp[1]);
  // CHECK: __spirv_ocl_sqrtd
  rd[idx++] = std::sqrt(dp[0]);
  // CHECK: __spirv_ocl_sqrtd
  rd[idx++] = std::sqrt(ip[0]);

  // CHECK: __spirv_ocl_cbrtf
  rf[idx++] = std::cbrt(fp[0]);
  // CHECK: __spirv_ocl_cbrtf
  rf[idx++] = std::cbrtf(fp[1]);
  // CHECK: __spirv_ocl_cbrtd
  rd[idx++] = std::cbrt(dp[0]);
  // CHECK: __spirv_ocl_cbrtd
  rd[idx++] = std::cbrt(ip[0]);

  // CHECK: __spirv_ocl_hypotff
  rf[idx++] = std::hypot(fp[0], fp[1]);
  // CHECK: __spirv_ocl_hypotff
  rf[idx++] = std::hypotf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_hypotdd
  rd[idx++] = std::hypot(dp[0], dp[1]);
  // CHECK: __spirv_ocl_hypotdd
  rd[idx++] = std::hypot(ip[0], fp[1]);
  // CHECK: __spirv_ocl_hypotdd
  rd[idx++] = std::hypot(dp[0], fp[1]);

  // CHECK: __spirv_ocl_sinf
  rf[idx++] = std::sin(fp[0]);
  // CHECK: __spirv_ocl_sinf
  rf[idx++] = std::sinf(fp[1]);
  // CHECK: __spirv_ocl_sind
  rd[idx++] = std::sin(dp[0]);
  // CHECK: __spirv_ocl_sind
  rd[idx++] = std::sin(ip[0]);

  // CHECK: __spirv_ocl_cosf
  rf[idx++] = std::cos(fp[0]);
  // CHECK: __spirv_ocl_cosf
  rf[idx++] = std::cosf(fp[1]);
  // CHECK: __spirv_ocl_cosd
  rd[idx++] = std::cos(dp[0]);
  // CHECK: __spirv_ocl_cosd
  rd[idx++] = std::cos(ip[0]);

  // CHECK: __spirv_ocl_tanf
  rf[idx++] = std::tan(fp[0]);
  // CHECK: __spirv_ocl_tanf
  rf[idx++] = std::tanf(fp[1]);
  // CHECK: __spirv_ocl_tand
  rd[idx++] = std::tan(dp[0]);
  // CHECK: __spirv_ocl_tand
  rd[idx++] = std::tan(ip[0]);

  // CHECK: __spirv_ocl_asinf
  rf[idx++] = std::asin(fp[0]);
  // CHECK: __spirv_ocl_asinf
  rf[idx++] = std::asinf(fp[1]);
  // CHECK: __spirv_ocl_asind
  rd[idx++] = std::asin(dp[0]);
  // CHECK: __spirv_ocl_asind
  rd[idx++] = std::asin(ip[0]);

  // CHECK: __spirv_ocl_acosf
  rf[idx++] = std::acos(fp[0]);
  // CHECK: __spirv_ocl_acosf
  rf[idx++] = std::acosf(fp[1]);
  // CHECK: __spirv_ocl_acosd
  rd[idx++] = std::acos(dp[0]);
  // CHECK: __spirv_ocl_acosd
  rd[idx++] = std::acos(ip[0]);

  // CHECK: __spirv_ocl_atanf
  rf[idx++] = std::atan(fp[0]);
  // CHECK: __spirv_ocl_atanf
  rf[idx++] = std::atanf(fp[1]);
  // CHECK: __spirv_ocl_atand
  rd[idx++] = std::atan(dp[0]);
  // CHECK: __spirv_ocl_atand
  rd[idx++] = std::atan(ip[0]);

  // CHECK: __spirv_ocl_atan2ff
  rf[idx++] = std::atan2(fp[0], fp[1]);
  // CHECK: __spirv_ocl_atan2ff
  rf[idx++] = std::atan2f(fp[2], fp[1]);
  // CHECK: __spirv_ocl_atan2dd
  rd[idx++] = std::atan2(dp[0], dp[1]);
  // CHECK: __spirv_ocl_atan2dd
  rd[idx++] = std::atan2(ip[0], fp[1]);
  // CHECK: __spirv_ocl_atan2dd
  rd[idx++] = std::atan2(dp[0], fp[1]);

  // CHECK: __spirv_ocl_sinhf
  rf[idx++] = std::sinh(fp[0]);
  // CHECK: __spirv_ocl_sinhf
  rf[idx++] = std::sinhf(fp[1]);
  // CHECK: __spirv_ocl_sinhd
  rd[idx++] = std::sinh(dp[0]);
  // CHECK: __spirv_ocl_sinhd
  rd[idx++] = std::sinh(ip[0]);

  // CHECK: __spirv_ocl_coshf
  rf[idx++] = std::cosh(fp[0]);
  // CHECK: __spirv_ocl_coshf
  rf[idx++] = std::coshf(fp[1]);
  // CHECK: __spirv_ocl_coshd
  rd[idx++] = std::cosh(dp[0]);
  // CHECK: __spirv_ocl_coshd
  rd[idx++] = std::cosh(ip[0]);

  // CHECK: __spirv_ocl_tanhf
  rf[idx++] = std::tanh(fp[0]);
  // CHECK: __spirv_ocl_tanhf
  rf[idx++] = std::tanhf(fp[1]);
  // CHECK: __spirv_ocl_tanhd
  rd[idx++] = std::tanh(dp[0]);
  // CHECK: __spirv_ocl_tanhd
  rd[idx++] = std::tanh(ip[0]);

  // CHECK: __spirv_ocl_asinhf
  rf[idx++] = std::asinh(fp[0]);
  // CHECK: __spirv_ocl_asinhf
  rf[idx++] = std::asinhf(fp[1]);
  // CHECK: __spirv_ocl_asinhd
  rd[idx++] = std::asinh(dp[0]);
  // CHECK: __spirv_ocl_asinhd
  rd[idx++] = std::asinh(ip[0]);

  // CHECK: __spirv_ocl_acoshf
  rf[idx++] = std::acosh(fp[0]);
  // CHECK: __spirv_ocl_acoshf
  rf[idx++] = std::acoshf(fp[1]);
  // CHECK: __spirv_ocl_acoshd
  rd[idx++] = std::acosh(dp[0]);
  // CHECK: __spirv_ocl_acoshd
  rd[idx++] = std::acosh(ip[0]);

  // CHECK: __spirv_ocl_atanhf
  rf[idx++] = std::atanh(fp[0]);
  // CHECK: __spirv_ocl_atanhf
  rf[idx++] = std::atanhf(fp[1]);
  // CHECK: __spirv_ocl_atanhd
  rd[idx++] = std::atanh(dp[0]);
  // CHECK: __spirv_ocl_atanhd
  rd[idx++] = std::atanh(ip[0]);

  // CHECK: __spirv_ocl_erff
  rf[idx++] = std::erf(fp[0]);
  // CHECK: __spirv_ocl_erff
  rf[idx++] = std::erff(fp[1]);
  // CHECK: __spirv_ocl_erfd
  rd[idx++] = std::erf(dp[0]);
  // CHECK: __spirv_ocl_erfd
  rd[idx++] = std::erf(ip[0]);

  // CHECK: __spirv_ocl_erfcf
  rf[idx++] = std::erfc(fp[0]);
  // CHECK: __spirv_ocl_erfcf
  rf[idx++] = std::erfcf(fp[1]);
  // CHECK: __spirv_ocl_erfcd
  rd[idx++] = std::erfc(dp[0]);
  // CHECK: __spirv_ocl_erfcd
  rd[idx++] = std::erfc(ip[0]);

  // CHECK: __spirv_ocl_tgammaf
  rf[idx++] = std::tgamma(fp[0]);
  // CHECK: __spirv_ocl_tgammaf
  rf[idx++] = std::tgammaf(fp[1]);
  // CHECK: __spirv_ocl_tgammad
  rd[idx++] = std::tgamma(dp[0]);
  // CHECK: __spirv_ocl_tgammad
  rd[idx++] = std::tgamma(ip[0]);

  // CHECK: __spirv_ocl_lgammaf
  rf[idx++] = std::lgamma(fp[0]);
  // CHECK: __spirv_ocl_lgammaf
  rf[idx++] = std::lgammaf(fp[1]);
  // CHECK: __spirv_ocl_lgammad
  rd[idx++] = std::lgamma(dp[0]);
  // CHECK: __spirv_ocl_lgammad
  rd[idx++] = std::lgamma(ip[0]);

  // CHECK: __spirv_ocl_ceilf
  rf[idx++] = std::ceil(fp[0]);
  // CHECK: __spirv_ocl_ceilf
  rf[idx++] = std::ceilf(fp[1]);
  // CHECK: __spirv_ocl_ceild
  rd[idx++] = std::ceil(dp[0]);
  // CHECK: __spirv_ocl_ceild
  rd[idx++] = std::ceil(ip[0]);

  // CHECK: __spirv_ocl_floorf
  rf[idx++] = std::floor(fp[0]);
  // CHECK: __spirv_ocl_floorf
  rf[idx++] = std::floorf(fp[1]);
  // CHECK: __spirv_ocl_floord
  rd[idx++] = std::floor(dp[0]);
  // CHECK: __spirv_ocl_floord
  rd[idx++] = std::floor(ip[0]);

  // CHECK: __spirv_ocl_truncf
  rf[idx++] = std::trunc(fp[0]);
  // CHECK: __spirv_ocl_truncf
  rf[idx++] = std::truncf(fp[1]);
  // CHECK: __spirv_ocl_truncd
  rd[idx++] = std::trunc(dp[0]);
  // CHECK: __spirv_ocl_truncd
  rd[idx++] = std::trunc(ip[0]);

  // CHECK: __spirv_ocl_roundf
  rf[idx++] = std::round(fp[0]);
  // CHECK: __spirv_ocl_roundf
  rf[idx++] = std::roundf(fp[1]);
  // CHECK: __spirv_ocl_roundd
  rd[idx++] = std::round(dp[0]);
  // CHECK: __spirv_ocl_roundd
  rd[idx++] = std::round(ip[0]);

  // CHECK: __spirv_ocl_rintf
  rf[idx++] = std::rint(fp[0]);
  // CHECK: __spirv_ocl_rintf
  rf[idx++] = std::rintf(fp[1]);
  // CHECK: __spirv_ocl_rintd
  rd[idx++] = std::rint(dp[0]);
  // CHECK: __spirv_ocl_rintd
  rd[idx++] = std::rint(ip[0]);

  // CHECK: __spirv_ocl_frexpf
  rf[idx++] = std::frexp(fp[0], ip);
  // CHECK: __spirv_ocl_frexpf
  rf[idx++] = std::frexpf(fp[1], ip);
  // CHECK: __spirv_ocl_frexpd
  rd[idx++] = std::frexp(dp[0], ip);
  // CHECK: __spirv_ocl_frexpd
  rd[idx++] = std::frexp(ip[0], ip);

  // CHECK: __spirv_ocl_ldexpf
  rf[idx++] = std::ldexp(fp[0], ip[0]);
  // CHECK: __spirv_ocl_ldexpf
  rf[idx++] = std::ldexpf(fp[1], ip[0]);
  // CHECK: __spirv_ocl_ldexpd
  rd[idx++] = std::ldexp(dp[0], ip[0]);
  // CHECK: __spirv_ocl_ldexpd
  rd[idx++] = std::ldexp(ip[0], ip[0]);

  // CHECK: __spirv_ocl_modff
  rf[idx++] = std::modf(fp[0], fp);
  // CHECK: __spirv_ocl_modff
  rf[idx++] = std::modff(fp[1], fp);
  // CHECK: __spirv_ocl_modfd
  rd[idx++] = std::modf(dp[0], dp);
  // CHECK: __spirv_ocl_modfd
  rd[idx++] = std::modf(ip[0], dp);

  // CHECK: __spirv_ocl_ldexpf
  rf[idx++] = std::scalbn(fp[0], ip[0]);
  // CHECK: __spirv_ocl_ldexpf
  rf[idx++] = std::scalbnf(fp[1], ip[0]);
  // CHECK: __spirv_ocl_ldexpd
  rd[idx++] = std::scalbn(dp[0], ip[0]);
  // CHECK: __spirv_ocl_ldexpd
  rd[idx++] = std::scalbn(ip[0], ip[0]);

  // CHECK: __spirv_ocl_ilogbf
  ri[idx++] = std::ilogb(fp[0]);
  // CHECK: __spirv_ocl_ilogbf
  ri[idx++] = std::ilogbf(fp[1]);
  // CHECK: __spirv_ocl_ilogbd
  ri[idx++] = std::ilogb(dp[0]);
  // CHECK: __spirv_ocl_ilogbd
  ri[idx++] = std::ilogb(ip[0]);

  // CHECK: __spirv_ocl_logbf
  rf[idx++] = std::logb(fp[0]);
  // CHECK: __spirv_ocl_logbf
  rf[idx++] = std::logbf(fp[1]);
  // CHECK: __spirv_ocl_logbd
  rd[idx++] = std::logb(dp[0]);
  // CHECK: __spirv_ocl_logbd
  rd[idx++] = std::logb(ip[0]);

  // CHECK: __spirv_ocl_nextafterf
  rf[idx++] = std::nextafter(fp[0], fp[1]);
  // CHECK: __spirv_ocl_nextafterf
  rf[idx++] = std::nextafterf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_nextafterd
  rd[idx++] = std::nextafter(dp[0], dp[1]);
  // CHECK: __spirv_ocl_nextafterd
  rd[idx++] = std::nextafter(ip[0], fp[1]);
  // CHECK: __spirv_ocl_nextafterd
  rd[idx++] = std::nextafter(dp[0], fp[1]);

  // CHECK: __spirv_ocl_copysignf
  rf[idx++] = std::copysign(fp[0], fp[1]);
  // CHECK: __spirv_ocl_copysignf
  rf[idx++] = std::copysignf(fp[2], fp[1]);
  // CHECK: __spirv_ocl_copysignd
  rd[idx++] = std::copysign(dp[0], dp[1]);
  // CHECK: __spirv_ocl_copysignd
  rd[idx++] = std::copysign(ip[0], fp[1]);
  // CHECK: __spirv_ocl_copysignd
  rd[idx++] = std::copysign(dp[0], fp[1]);
}
