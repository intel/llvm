//==------------- __sycl_complex_impl.hpp ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_COMPLEX_IMPL_HPP__
#define __SYCL_COMPLEX_IMPL_HPP__

// This header defines device-side overloads of <complex> functions.

#ifdef __SYCL_DEVICE_ONLY__

#include <limits>

// The 'sycl_device_only' attribute enables device-side overloading.
#define __SYCL_DEVICE __attribute__((sycl_device_only, always_inline))
#define __SYCL_DEVICE_C                                                        \
  extern "C" __attribute__((sycl_device_only, always_inline))

// TODO: This needs to be more robust.
// clang doesn't recognize the c11 __SYCL_CMPLX macro, but it does have
//   its own syntax extension for initializing a complex as a struct.
#ifndef __SYCL_CMPLXF
#define __SYCL_CMPLXF(r, i) ((float __complex__){(float)(r), (float)(i)})
#endif
#ifndef __SYCL_CMPLX
#define __SYCL_CMPLX(r, i) ((double __complex__){(double)(r), (double)(i)})
#endif

__SYCL_DEVICE_C
float cimagf(float __complex__ z) { return __imag__(z); }
__SYCL_DEVICE_C
double cimag(double __complex__ z) { return __imag__(z); }

__SYCL_DEVICE_C
float crealf(float __complex__ z) { return __real__(z); }
__SYCL_DEVICE_C
double creal(double __complex__ z) { return __real__(z); }

// __mulsc3
// Returns: the product of a + ib and c + id.
__SYCL_DEVICE_C
float __complex__ __mulsc3(float __a, float __b, float __c, float __d) {
  float __ac = __a * __c;
  float __bd = __b * __d;
  float __ad = __a * __d;
  float __bc = __b * __c;
  float __complex__ z;
  z = __SYCL_CMPLXF((__ac - __bd), (__ad + __bc));
  if (__spirv_IsNan(crealf(z)) && __spirv_IsNan(cimagf(z))) {
    int __recalc = 0;
    if (__spirv_IsInf(__a) || __spirv_IsInf(__b)) {
      __a = __spirv_ocl_copysign(__spirv_IsInf(__a) ? 1.0f : 0.0f, __a);
      __b = __spirv_ocl_copysign(__spirv_IsInf(__b) ? 1.0f : 0.0f, __b);
      if (__spirv_IsNan(__c))
        __c = __spirv_ocl_copysign(0.0f, __c);
      if (__spirv_IsNan(__d))
        __d = __spirv_ocl_copysign(0.0f, __d);
      __recalc = 1;
    }
    if (__spirv_IsInf(__c) || __spirv_IsInf(__d)) {
      __c = __spirv_ocl_copysign(__spirv_IsInf(__c) ? 1.0f : 0.0f, __c);
      __d = __spirv_ocl_copysign(__spirv_IsInf(__d) ? 1.0f : 0.0f, __d);
      if (__spirv_IsNan(__a))
        __a = __spirv_ocl_copysign(0.0f, __a);
      if (__spirv_IsNan(__b))
        __b = __spirv_ocl_copysign(0.0f, __b);
      __recalc = 1;
    }
    if (!__recalc && (__spirv_IsInf(__ac) || __spirv_IsInf(__bd) ||
                      __spirv_IsInf(__ad) || __spirv_IsInf(__bc))) {
      if (__spirv_IsNan(__a))
        __a = __spirv_ocl_copysign(0.0f, __a);
      if (__spirv_IsNan(__b))
        __b = __spirv_ocl_copysign(0.0f, __b);
      if (__spirv_IsNan(__c))
        __c = __spirv_ocl_copysign(0.0f, __c);
      if (__spirv_IsNan(__d))
        __d = __spirv_ocl_copysign(0.0f, __d);
      __recalc = 1.0f;
    }
    if (__recalc) {
      z = __SYCL_CMPLXF(
          (std::numeric_limits<float>::infinity() * (__a * __c - __b * __d)),
          (std::numeric_limits<float>::infinity() * (__a * __d + __b * __c)));
    }
  }
  return z;
}
// __muldc3
// Returns: the product of a + ib and c + id
__SYCL_DEVICE_C
double __complex__ __muldc3(double __a, double __b, double __c, double __d) {
  double __ac = __a * __c;
  double __bd = __b * __d;
  double __ad = __a * __d;
  double __bc = __b * __c;
  double __complex__ z;
  z = __SYCL_CMPLX((__ac - __bd), (__ad + __bc));
  if (__spirv_IsNan(creal(z)) && __spirv_IsNan(cimag(z))) {
    int __recalc = 0;
    if (__spirv_IsInf(__a) || __spirv_IsInf(__b)) {
      __a = __spirv_ocl_copysign(__spirv_IsInf(__a) ? 1.0 : 0.0, __a);
      __b = __spirv_ocl_copysign(__spirv_IsInf(__b) ? 1.0 : 0.0, __b);
      if (__spirv_IsNan(__c))
        __c = __spirv_ocl_copysign(0.0, __c);
      if (__spirv_IsNan(__d))
        __d = __spirv_ocl_copysign(0.0, __d);
      __recalc = 1;
    }
    if (__spirv_IsInf(__c) || __spirv_IsInf(__d)) {
      __c = __spirv_ocl_copysign(__spirv_IsInf(__c) ? 1.0 : 0.0, __c);
      __d = __spirv_ocl_copysign(__spirv_IsInf(__d) ? 1.0 : 0.0, __d);
      if (__spirv_IsNan(__a))
        __a = __spirv_ocl_copysign(0.0, __a);
      if (__spirv_IsNan(__b))
        __b = __spirv_ocl_copysign(0.0, __b);
      __recalc = 1;
    }
    if (!__recalc && (__spirv_IsInf(__ac) || __spirv_IsInf(__bd) ||
                      __spirv_IsInf(__ad) || __spirv_IsInf(__bc))) {
      if (__spirv_IsNan(__a))
        __a = __spirv_ocl_copysign(0.0, __a);
      if (__spirv_IsNan(__b))
        __b = __spirv_ocl_copysign(0.0, __b);
      if (__spirv_IsNan(__c))
        __c = __spirv_ocl_copysign(0.0, __c);
      if (__spirv_IsNan(__d))
        __d = __spirv_ocl_copysign(0.0, __d);
      __recalc = 1.0;
    }
    if (__recalc) {
      z = __SYCL_CMPLX(
          (std::numeric_limits<float>::infinity() * (__a * __c - __b * __d)),
          (std::numeric_limits<float>::infinity() * (__a * __d + __b * __c)));
    }
  }
  return z;
}

// __divsc3
// Returns: the quotient of (a + ib) / (c + id).
// FIXME: divsc3/divdc3 have overflow issue when dealing with large number.
// And this overflow issue is from libc++/compiler-rt's implementation.
__SYCL_DEVICE_C
float __complex__ __divsc3(float __a, float __b, float __c, float __d) {
  int __ilogbw = 0;
  float __logbw = __spirv_ocl_logb(
      __spirv_ocl_fmax(__spirv_ocl_fabs(__c), __spirv_ocl_fabs(__d)));
  if (__spirv_IsFinite(__logbw)) {
    __ilogbw = (int)__logbw;
    __c = __spirv_ocl_ldexp(__c, -__ilogbw);
    __d = __spirv_ocl_ldexp(__d, -__ilogbw);
  }
  float __denom = __c * __c + __d * __d;
  float __complex__ z;
  float z_real =
      __spirv_ocl_ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
  float z_imag =
      __spirv_ocl_ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);
  z = __SYCL_CMPLXF(z_real, z_imag);
  if (__spirv_IsNan(z_real) && __spirv_IsNan(z_imag)) {
    if ((__denom == 0.0f) && (!__spirv_IsNan(__a) || !__spirv_IsNan(__b))) {
      z_real =
          __spirv_ocl_copysign(std::numeric_limits<float>::infinity(), __c) *
          __a;
      z_imag =
          __spirv_ocl_copysign(std::numeric_limits<float>::infinity(), __c) *
          __b;
      z = __SYCL_CMPLXF(z_real, z_imag);
    } else if ((__spirv_IsInf(__a) || __spirv_IsInf(__b)) &&
               __spirv_IsFinite(__c) && __spirv_IsFinite(__d)) {
      __a = __spirv_ocl_copysign(__spirv_IsInf(__a) ? 1.0f : 0.0f, __a);
      __b = __spirv_ocl_copysign(__spirv_IsInf(__b) ? 1.0f : 0.0f, __b);
      z_real = std::numeric_limits<float>::infinity() * (__a * __c + __b * __d);
      z_imag = std::numeric_limits<float>::infinity() * (__b * __c - __a * __d);
      z = __SYCL_CMPLXF(z_real, z_imag);
    } else if (__spirv_IsInf(__logbw) && __logbw > 0.0f &&
               __spirv_IsFinite(__a) && __spirv_IsFinite(__b)) {
      __c = __spirv_ocl_copysign(__spirv_IsInf(__c) ? 1.0f : 0.0f, __c);
      __d = __spirv_ocl_copysign(__spirv_IsInf(__d) ? 1.0f : 0.0f, __d);
      z_real = 0.0f * (__a * __c + __b * __d);
      z_imag = 0.0f * (__b * __c - __a * __d);
      z = __SYCL_CMPLXF(z_real, z_imag);
    }
  }
  return z;
}
// __divdc3
// Returns: the quotient of (a + ib) / (c + id).
__SYCL_DEVICE_C
double __complex__ __divdc3(double __a, double __b, double __c, double __d) {
  int __ilogbw = 0;
  double __logbw = __spirv_ocl_logb(
      __spirv_ocl_fmax(__spirv_ocl_fabs(__c), __spirv_ocl_fabs(__d)));
  if (__spirv_IsFinite(__logbw)) {
    __ilogbw = (int)__logbw;
    __c = __spirv_ocl_ldexp(__c, -__ilogbw);
    __d = __spirv_ocl_ldexp(__d, -__ilogbw);
  }
  double __denom = __c * __c + __d * __d;
  double __complex__ z;
  double z_real =
      __spirv_ocl_ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
  double z_imag =
      __spirv_ocl_ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);
  z = __SYCL_CMPLX(z_real, z_imag);
  if (__spirv_IsNan(z_real) && __spirv_IsNan(z_imag)) {
    if ((__denom == 0.0) && (!__spirv_IsNan(__a) || !__spirv_IsNan(__b))) {
      z_real =
          __spirv_ocl_copysign(std::numeric_limits<double>::infinity(), __c) *
          __a;
      z_imag =
          __spirv_ocl_copysign(std::numeric_limits<double>::infinity(), __c) *
          __b;
      z = __SYCL_CMPLX(z_real, z_imag);
    } else if ((__spirv_IsInf(__a) || __spirv_IsInf(__b)) &&
               __spirv_IsFinite(__c) && __spirv_IsFinite(__d)) {
      __a = __spirv_ocl_copysign(__spirv_IsInf(__a) ? 1.0 : 0.0, __a);
      __b = __spirv_ocl_copysign(__spirv_IsInf(__b) ? 1.0 : 0.0, __b);
      z_real = std::numeric_limits<float>::infinity() * (__a * __c + __b * __d);
      z_imag = std::numeric_limits<float>::infinity() * (__b * __c - __a * __d);
      z = __SYCL_CMPLX(z_real, z_imag);
    } else if (__spirv_IsInf(__logbw) && __logbw > 0.0 &&
               __spirv_IsFinite(__a) && __spirv_IsFinite(__b)) {
      __c = __spirv_ocl_copysign(__spirv_IsInf(__c) ? 1.0 : 0.0, __c);
      __d = __spirv_ocl_copysign(__spirv_IsInf(__d) ? 1.0 : 0.0, __d);
      z_real = 0.0 * (__a * __c + __b * __d);
      z_imag = 0.0 * (__b * __c - __a * __d);
      z = __SYCL_CMPLX(z_real, z_imag);
    }
  }
  return z;
}

__SYCL_DEVICE_C
float cargf(float __complex__ z) {
  return __spirv_ocl_atan2(cimagf(z), crealf(z));
}
__SYCL_DEVICE_C
double carg(double __complex__ z) {
  return __spirv_ocl_atan2(cimag(z), creal(z));
}

__SYCL_DEVICE_C
float cabsf(float __complex__ z) {
  return __spirv_ocl_hypot(crealf(z), cimagf(z));
}
__SYCL_DEVICE_C
double cabs(double __complex__ z) {
  return __spirv_ocl_hypot(creal(z), cimag(z));
}

__SYCL_DEVICE_C
float __complex__ cprojf(float __complex__ z) {
  float __complex__ r = z;
  if (__spirv_IsInf(crealf(z)) || __spirv_IsInf(cimagf(z)))
    r = __SYCL_CMPLXF(std::numeric_limits<float>::infinity(),
                      __spirv_ocl_copysign(0.0f, cimagf(z)));
  return r;
}
__SYCL_DEVICE_C
double __complex__ cproj(double __complex__ z) {
  double __complex__ r = z;
  if (__spirv_IsInf(creal(z)) || __spirv_IsInf(cimag(z)))
    r = __SYCL_CMPLX(std::numeric_limits<float>::infinity(),
                     __spirv_ocl_copysign(0.0, cimag(z)));
  return r;
}

__SYCL_DEVICE_C
float __complex__ cexpf(float __complex__ z) {
  float z_imag = cimagf(z);
  float z_real = crealf(z);
  if (z_imag == 0) {
    return __SYCL_CMPLXF(__spirv_ocl_exp(z_real),
                         __spirv_ocl_copysign(0.f, z_imag));
  }

  if (__spirv_IsInf(z_real)) {
    if (z_real < 0.f) {
      if (!__spirv_IsFinite(z_imag))
        z_imag = 1.0f;
    } else if (__spirv_IsNan(z_imag)) {
      return z;
    } else if (z_imag == 0.f || !__spirv_IsFinite(z_imag)) {
      if (__spirv_IsInf(z_imag))
        return __SYCL_CMPLXF(z_real, std::numeric_limits<float>::quiet_NaN());
    }
  }

  float e = __spirv_ocl_exp(z_real);
  return __SYCL_CMPLXF(e * __spirv_ocl_cos(z_imag),
                       e * __spirv_ocl_sin(z_imag));
}
__SYCL_DEVICE_C
double __complex__ cexp(double __complex__ z) {
  double z_imag = cimag(z);
  double z_real = creal(z);
  if (__spirv_IsInf(z_real)) {
    if (z_real < 0.0) {
      if (!__spirv_IsFinite(z_imag))
        z_imag = 1.0;
    } else if (z_imag == 0.0 || !__spirv_IsFinite(z_imag)) {
      if (__spirv_IsInf(z_imag))
        z_imag = std::numeric_limits<float>::quiet_NaN();
      return __SYCL_CMPLX(z_real, z_imag);
    }
  } else if (__spirv_IsNan(z_real)) {
    if (z_imag == 0.0)
      return z;
    return __SYCL_CMPLX(std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN());
  } else if (__spirv_IsFinite(z_real) &&
             (__spirv_IsNan(z_imag) || __spirv_IsInf(z_imag))) {
    return __SYCL_CMPLX(std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN());
  }
  double __e = __spirv_ocl_exp(z_real);
  double ret_real = __e * __spirv_ocl_cos(z_imag);
  double ret_imag = __e * __spirv_ocl_sin(z_imag);

  if (__spirv_IsNan(ret_real))
    ret_real = 0.;
  if (__spirv_IsNan(ret_imag))
    ret_imag = 0.;
  return __SYCL_CMPLX(ret_real, ret_imag);
}

__SYCL_DEVICE_C
float __complex__ clogf(float __complex__ z) {
  return __SYCL_CMPLXF(__spirv_ocl_log(cabsf(z)), cargf(z));
}
__SYCL_DEVICE_C
double __complex__ clog(double __complex__ z) {
  return __SYCL_CMPLX(__spirv_ocl_log(cabs(z)), carg(z));
}

__SYCL_DEVICE_C
float __complex__ clog10f(float __complex__ z) {
  float __complex__ lz = clogf(z);
  float __complex__ d = __mulsc3(crealf(lz), cimagf(lz), 0x1.bcb7bp-2f, 0.f);
  return d;
}
__SYCL_DEVICE_C
double __complex__ clog10(double __complex__ z) {
  double __complex__ lz = clog(z);
  double __complex__ d =
      __muldc3(creal(lz), cimag(lz), 0x1.bcb7b1526e50dp-2, 0.);
  return d;
}

__SYCL_DEVICE_C
float __complex__ cpowf(float __complex__ x, float __complex__ y) {
  float __complex__ t = clogf(x);
  float __complex__ w = __mulsc3(crealf(y), cimagf(y), crealf(t), cimagf(t));
  return cexpf(w);
}
__SYCL_DEVICE_C
double __complex__ cpow(double __complex__ x, double __complex__ y) {
  double __complex__ t = clog(x);
  double __complex__ w = __muldc3(creal(y), cimag(y), creal(t), cimag(t));
  return cexp(w);
}

__SYCL_DEVICE_C
float __complex__ cpolarf(float rho, float theta) {
  if (__spirv_IsNan(rho) || __spirv_SignBitSet(rho))
    return __SYCL_CMPLXF(std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN());
  if (__spirv_IsNan(theta)) {
    if (__spirv_IsInf(rho))
      return __SYCL_CMPLXF(rho, theta);
    return __SYCL_CMPLXF(theta, theta);
  }
  if (__spirv_IsInf(theta)) {
    if (__spirv_IsInf(rho))
      return __SYCL_CMPLXF(rho, std::numeric_limits<float>::quiet_NaN());
    return __SYCL_CMPLXF(std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN());
  }
  float x = rho * __spirv_ocl_cos(theta);
  if (__spirv_IsNan(x))
    x = 0;
  float y = rho * __spirv_ocl_sin(theta);
  if (__spirv_IsNan(y))
    y = 0;
  return __SYCL_CMPLXF(x, y);
}
__SYCL_DEVICE_C
double __complex__ cpolar(double rho, double theta) {
  if (__spirv_IsNan(rho) || __spirv_SignBitSet(rho))
    return __SYCL_CMPLX(std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN());
  if (__spirv_IsNan(theta)) {
    if (__spirv_IsInf(rho))
      return __SYCL_CMPLX(rho, theta);
    return __SYCL_CMPLX(theta, theta);
  }
  if (__spirv_IsInf(theta)) {
    if (__spirv_IsInf(rho))
      return __SYCL_CMPLX(rho, std::numeric_limits<float>::quiet_NaN());
    return __SYCL_CMPLX(std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN());
  }
  double x = rho * __spirv_ocl_cos(theta);
  if (__spirv_IsNan(x))
    x = 0;
  double y = rho * __spirv_ocl_sin(theta);
  if (__spirv_IsNan(y))
    y = 0;
  return __SYCL_CMPLX(x, y);
}

__SYCL_DEVICE_C
float __complex__ csqrtf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLXF(std::numeric_limits<float>::infinity(), z_imag);
  if (__spirv_IsInf(z_real)) {
    if (z_real > 0.0f)
      return __SYCL_CMPLXF(z_real, __spirv_IsNan(z_imag)
                                       ? z_imag
                                       : __spirv_ocl_copysign(0.0f, z_imag));
    return __SYCL_CMPLXF(__spirv_IsNan(z_imag) ? z_imag : 0.0f,
                         __spirv_ocl_copysign(z_real, z_imag));
  }
  return cpolarf(__spirv_ocl_sqrt(cabsf(z)), cargf(z) / 2.0f);
}
__SYCL_DEVICE_C
double __complex__ csqrt(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLX(std::numeric_limits<float>::infinity(), z_imag);
  if (__spirv_IsInf(z_real)) {
    if (z_real > 0.0)
      return __SYCL_CMPLX(z_real, __spirv_IsNan(z_imag)
                                      ? z_imag
                                      : __spirv_ocl_copysign(0.0, z_imag));
    return __SYCL_CMPLX(__spirv_IsNan(z_imag) ? z_imag : 0.0,
                        __spirv_ocl_copysign(z_real, z_imag));
  }
  return cpolar(__spirv_ocl_sqrt(cabs(z)), carg(z) / 2.0);
}

__SYCL_DEVICE_C
float __complex__ csinhf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  if (__spirv_IsInf(z_real) && !__spirv_IsFinite(z_imag))
    return __SYCL_CMPLXF(z_real, std::numeric_limits<float>::quiet_NaN());
  if (z_real == 0 && !__spirv_IsFinite(z_imag))
    return __SYCL_CMPLXF(z_real, std::numeric_limits<float>::quiet_NaN());
  if (z_imag == 0 && !__spirv_IsFinite(z_real))
    return z;
  return __SYCL_CMPLXF(__spirv_ocl_sinh(z_real) * __spirv_ocl_cos(z_imag),
                       __spirv_ocl_cosh(z_real) * __spirv_ocl_sin(z_imag));
}
__SYCL_DEVICE_C
double __complex__ csinh(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  if (__spirv_IsInf(z_real) && !__spirv_IsFinite(z_imag))
    return __SYCL_CMPLX(z_real, std::numeric_limits<float>::quiet_NaN());
  if (z_real == 0 && !__spirv_IsFinite(z_imag))
    return __SYCL_CMPLX(z_real, std::numeric_limits<float>::quiet_NaN());
  if (z_imag == 0 && !__spirv_IsFinite(z_real))
    return z;
  return __SYCL_CMPLX(__spirv_ocl_sinh(z_real) * __spirv_ocl_cos(z_imag),
                      __spirv_ocl_cosh(z_real) * __spirv_ocl_sin(z_imag));
}

__SYCL_DEVICE_C
float __complex__ ccoshf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  if (__spirv_IsInf(z_real) && !__spirv_IsFinite(z_imag))
    return __SYCL_CMPLXF(__spirv_ocl_fabs(z_real),
                         std::numeric_limits<float>::quiet_NaN());
  if (z_real == 0 && !__spirv_IsFinite(z_imag))
    return __SYCL_CMPLXF(std::numeric_limits<float>::quiet_NaN(), z_real);
  if (z_real == 0 && z_imag == 0)
    return __SYCL_CMPLXF(1.0f, z_imag);
  if (z_imag == 0 && !__spirv_IsFinite(z_real))
    return __SYCL_CMPLXF(__spirv_ocl_fabs(z_real), z_imag);
  return __SYCL_CMPLXF(__spirv_ocl_cosh(z_real) * __spirv_ocl_cos(z_imag),
                       __spirv_ocl_sinh(z_real) * __spirv_ocl_sin(z_imag));
}
__SYCL_DEVICE_C
double __complex__ ccosh(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  if (__spirv_IsInf(z_real) && !__spirv_IsFinite(z_imag))
    return __SYCL_CMPLX(__spirv_ocl_fabs(z_real),
                        std::numeric_limits<float>::quiet_NaN());
  if (z_real == 0 && !__spirv_IsFinite(z_imag))
    return __SYCL_CMPLX(std::numeric_limits<float>::quiet_NaN(), z_real);
  if (z_real == 0 && z_imag == 0)
    return __SYCL_CMPLX(1.0f, z_imag);
  if (z_imag == 0 && !__spirv_IsFinite(z_real))
    return __SYCL_CMPLX(__spirv_ocl_fabs(z_real), z_imag);
  return __SYCL_CMPLX(__spirv_ocl_cosh(z_real) * __spirv_ocl_cos(z_imag),
                      __spirv_ocl_sinh(z_real) * __spirv_ocl_sin(z_imag));
}

__SYCL_DEVICE_C
float __complex__ ctanhf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  if (__spirv_IsInf(z_real)) {
    if (!__spirv_IsFinite(z_imag))
      return __SYCL_CMPLXF(__spirv_ocl_copysign(1.0f, z_real), 0.0f);
    return __SYCL_CMPLXF(
        __spirv_ocl_copysign(1.0f, z_real),
        __spirv_ocl_copysign(0.0f, __spirv_ocl_sin(2.0f * z_imag)));
  }
  if (__spirv_IsNan(z_real) && z_imag == 0)
    return z;
  float __2r(2.0f * z_real);
  float __2i(2.0f * z_imag);
  float __d(__spirv_ocl_cosh(__2r) + __spirv_ocl_cos(__2i));
  float __2rsh(__spirv_ocl_sinh(__2r));
  if (__spirv_IsInf(__2rsh) && __spirv_IsInf(__d))
    return __SYCL_CMPLXF(((__2rsh > 0.0f) ? 1.0f : -1.0f),
                         ((__2i > 0.0f) ? 0.0f : -0.0f));
  return __SYCL_CMPLXF(__2rsh / __d, __spirv_ocl_sin(__2i) / __d);
}
__SYCL_DEVICE_C
double __complex__ ctanh(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  if (__spirv_IsInf(z_real)) {
    if (!__spirv_IsFinite(z_imag))
      return __SYCL_CMPLX(__spirv_ocl_copysign(1.0, z_real), 0.0);
    return __SYCL_CMPLX(
        __spirv_ocl_copysign(1.0, z_real),
        __spirv_ocl_copysign(0.0, __spirv_ocl_sin(2.0 * z_imag)));
  }
  if (__spirv_IsNan(z_real) && z_imag == 0)
    return z;
  double __2r(2.0 * z_real);
  double __2i(2.0 * z_imag);
  double __d(__spirv_ocl_cosh(__2r) + __spirv_ocl_cos(__2i));
  double __2rsh(__spirv_ocl_sinh(__2r));
  if (__spirv_IsInf(__2rsh) && __spirv_IsInf(__d))
    return __SYCL_CMPLX(((__2rsh > 0.0) ? 1.0 : -1.0),
                        ((__2i > 0.0) ? 0.0 : -0.0));
  return __SYCL_CMPLX(__2rsh / __d, __spirv_ocl_sin(__2i) / __d);
}

__SYCL_DEVICE_C
float __complex__ csinf(float __complex__ z) {
  float __complex__ w = csinhf(__SYCL_CMPLXF(-cimagf(z), crealf(z)));
  return __SYCL_CMPLXF(cimagf(w), -crealf(w));
}
__SYCL_DEVICE_C
double __complex__ csin(double __complex__ z) {
  double __complex__ w = csinh(__SYCL_CMPLX(-cimag(z), creal(z)));
  return __SYCL_CMPLX(cimag(w), -creal(w));
}

__SYCL_DEVICE_C
float __complex__ ccosf(float __complex__ z) {
  return ccoshf(__SYCL_CMPLXF(-cimagf(z), crealf(z)));
}
__SYCL_DEVICE_C
double __complex__ ccos(double __complex__ z) {
  return ccosh(__SYCL_CMPLX(-cimag(z), creal(z)));
}

__SYCL_DEVICE_C
float __complex__ ctanf(float __complex__ z) {
  float __complex__ w = ctanhf(__SYCL_CMPLXF(-cimagf(z), crealf(z)));
  return __SYCL_CMPLXF(cimagf(w), -crealf(w));
}
__SYCL_DEVICE_C
double __complex__ ctan(double __complex__ z) {
  double __complex__ w = ctanh(__SYCL_CMPLX(-cimag(z), creal(z)));
  return __SYCL_CMPLX(cimag(w), -creal(w));
}

__SYCL_DEVICE_C
float __complex__ __sqrf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  return __SYCL_CMPLXF((z_real + z_imag) * (z_real - z_imag),
                       2.0f * z_real * z_imag);
}
__SYCL_DEVICE_C
double __complex__ __sqr(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  return __SYCL_CMPLX((z_real + z_imag) * (z_real - z_imag),
                      2.0 * z_real * z_imag);
}

__SYCL_DEVICE_C
float __complex__ cacosf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  const float __pi(__spirv_ocl_atan2(+0.0f, -0.0f));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return __SYCL_CMPLXF(z_imag, z_real);
    if (__spirv_IsInf(z_imag)) {
      if (z_real < 0.0f)
        return __SYCL_CMPLXF(0.75f * __pi, -z_imag);
      return __SYCL_CMPLXF(0.25f * __pi, -z_imag);
    }
    if (z_real < 0.0f)
      return __SYCL_CMPLXF(__pi, __spirv_SignBitSet(z_imag) ? -z_real : z_real);
    return __SYCL_CMPLXF(0.0f, __spirv_SignBitSet(z_imag) ? z_real : -z_real);
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return __SYCL_CMPLXF(z_real, -z_imag);
    return __SYCL_CMPLXF(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLXF(__pi / 2.0f, -z_imag);
  if (z_real == 0 && (z_imag == 0 || __spirv_IsNan(z_imag)))
    return __SYCL_CMPLXF(__pi / 2.0f, -z_imag);
  float __complex__ w = clogf(z + csqrtf(__sqrf(z) - 1.0f));
  if (__spirv_SignBitSet(z_imag))
    return __SYCL_CMPLXF(__spirv_ocl_fabs(cimagf(w)),
                         __spirv_ocl_fabs(crealf(w)));
  return __SYCL_CMPLXF(__spirv_ocl_fabs(cimagf(w)),
                       -__spirv_ocl_fabs(crealf(w)));
}
__SYCL_DEVICE_C
double __complex__ cacos(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  const double __pi(__spirv_ocl_atan2(+0.0, -0.0));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return __SYCL_CMPLX(z_imag, z_real);
    if (__spirv_IsInf(z_imag)) {
      if (z_real < 0.0)
        return __SYCL_CMPLX(0.75 * __pi, -z_imag);
      return __SYCL_CMPLX(0.25 * __pi, -z_imag);
    }
    if (z_real < 0.0)
      return __SYCL_CMPLX(__pi, __spirv_SignBitSet(z_imag) ? -z_real : z_real);
    return __SYCL_CMPLX(0.0f, __spirv_SignBitSet(z_imag) ? z_real : -z_real);
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return __SYCL_CMPLX(z_real, -z_imag);
    return __SYCL_CMPLX(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLX(__pi / 2.0, -z_imag);
  if (z_real == 0 && (z_imag == 0 || __spirv_IsNan(z_imag)))
    return __SYCL_CMPLX(__pi / 2.0, -z_imag);
  double __complex__ w = clog(z + csqrt(__sqr(z) - 1.0));
  if (__spirv_SignBitSet(z_imag))
    return __SYCL_CMPLX(__spirv_ocl_fabs(cimag(w)), __spirv_ocl_fabs(creal(w)));
  return __SYCL_CMPLX(__spirv_ocl_fabs(cimag(w)), -__spirv_ocl_fabs(creal(w)));
}

__SYCL_DEVICE_C
float __complex__ casinhf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  const float __pi(__spirv_ocl_atan2(+0.0f, -0.0f));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return z;
    if (__spirv_IsInf(z_imag))
      return __SYCL_CMPLXF(z_real, __spirv_ocl_copysign(__pi * 0.25f, z_imag));
    return __SYCL_CMPLXF(z_real, __spirv_ocl_copysign(0.0f, z_imag));
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return __SYCL_CMPLXF(z_imag, z_real);
    if (z_imag == 0)
      return z;
    return __SYCL_CMPLXF(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLXF(__spirv_ocl_copysign(z_imag, z_real),
                         __spirv_ocl_copysign(__pi / 2.0f, z_imag));
  float __complex__ w = clogf(z + csqrtf(__sqrf(z) + 1.0f));
  return __SYCL_CMPLXF(__spirv_ocl_copysign(crealf(w), z_real),
                       __spirv_ocl_copysign(cimagf(w), z_imag));
}
__SYCL_DEVICE_C
double __complex__ casinh(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  const double __pi(__spirv_ocl_atan2(+0.0, -0.0));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return z;
    if (__spirv_IsInf(z_imag))
      return __SYCL_CMPLX(z_real, __spirv_ocl_copysign(__pi * 0.25, z_imag));
    return __SYCL_CMPLX(z_real, __spirv_ocl_copysign(0.0, z_imag));
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return __SYCL_CMPLX(z_imag, z_real);
    if (z_imag == 0)
      return z;
    return __SYCL_CMPLX(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLX(__spirv_ocl_copysign(z_imag, z_real),
                        __spirv_ocl_copysign(__pi / 2.0, z_imag));
  double __complex__ w = clog(z + csqrt(__sqr(z) + 1.0));
  return __SYCL_CMPLX(__spirv_ocl_copysign(creal(w), z_real),
                      __spirv_ocl_copysign(cimag(w), z_imag));
}

__SYCL_DEVICE_C
float __complex__ casinf(float __complex__ z) {
  float __complex__ w = casinhf(__SYCL_CMPLXF(-cimagf(z), crealf(z)));
  return __SYCL_CMPLXF(cimagf(w), -crealf(w));
}
__SYCL_DEVICE_C
double __complex__ casin(double __complex__ z) {
  double __complex__ w = casinh(__SYCL_CMPLX(-cimag(z), creal(z)));
  return __SYCL_CMPLX(cimag(w), -creal(w));
}

__SYCL_DEVICE_C
float __complex__ cacoshf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  const float __pi(__spirv_ocl_atan2(+0.0f, -0.0f));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return __SYCL_CMPLXF(__spirv_ocl_fabs(z_real), z_imag);
    if (__spirv_IsInf(z_imag)) {
      if (z_real > 0)
        return __SYCL_CMPLXF(z_real,
                             __spirv_ocl_copysign(__pi * 0.25f, z_imag));
      else
        return __SYCL_CMPLXF(-z_real,
                             __spirv_ocl_copysign(__pi * 0.75f, z_imag));
    }
    if (z_real < 0)
      return __SYCL_CMPLXF(-z_real, __spirv_ocl_copysign(__pi, z_imag));
    return __SYCL_CMPLXF(z_real, __spirv_ocl_copysign(0.0f, z_imag));
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return __SYCL_CMPLXF(__spirv_ocl_fabs(z_imag), z_real);
    return __SYCL_CMPLXF(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLXF(__spirv_ocl_fabs(z_imag),
                         __spirv_ocl_copysign(__pi / 2.0f, z_imag));
  float __complex__ w = clogf(z + csqrtf(__sqrf(z) - 1.0f));
  return __SYCL_CMPLXF(__spirv_ocl_copysign(crealf(w), 0.0f),
                       __spirv_ocl_copysign(cimagf(w), z_imag));
}
__SYCL_DEVICE_C
double __complex__ cacosh(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  const double __pi(__spirv_ocl_atan2(+0.0, -0.0));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return __SYCL_CMPLX(__spirv_ocl_fabs(z_real), z_imag);
    if (__spirv_IsInf(z_imag)) {
      if (z_real > 0)
        return __SYCL_CMPLX(z_real, __spirv_ocl_copysign(__pi * 0.25f, z_imag));
      else
        return __SYCL_CMPLX(-z_real,
                            __spirv_ocl_copysign(__pi * 0.75f, z_imag));
    }
    if (z_real < 0)
      return __SYCL_CMPLX(-z_real, __spirv_ocl_copysign(__pi, z_imag));
    return __SYCL_CMPLX(z_real, __spirv_ocl_copysign(0.0, z_imag));
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return __SYCL_CMPLX(__spirv_ocl_fabs(z_imag), z_real);
    return __SYCL_CMPLX(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLX(__spirv_ocl_fabs(z_imag),
                        __spirv_ocl_copysign(__pi / 2.0, z_imag));
  double __complex__ w = clog(z + csqrt(__sqr(z) - 1.0));
  return __SYCL_CMPLX(__spirv_ocl_copysign(creal(w), 0.0),
                      __spirv_ocl_copysign(cimag(w), z_imag));
}

__SYCL_DEVICE_C
float __complex__ catanhf(float __complex__ z) {
  float z_real = crealf(z);
  float z_imag = cimagf(z);
  const float __pi(__spirv_ocl_atan2(+0.0f, -0.0f));
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLXF(__spirv_ocl_copysign(0.0f, z_real),
                         __spirv_ocl_copysign(__pi / 2.0f, z_imag));
  if (__spirv_IsNan(z_imag)) {
    if (__spirv_IsInf(z_real) || z_real == 0)
      return __SYCL_CMPLXF(__spirv_ocl_copysign(0.0f, z_real), z_imag);
    return __SYCL_CMPLXF(z_imag, z_imag);
  }
  if (__spirv_IsNan(z_real))
    return __SYCL_CMPLXF(z_real, z_real);
  if (__spirv_IsInf(z_real))
    return __SYCL_CMPLXF(__spirv_ocl_copysign(0.0f, z_real),
                         __spirv_ocl_copysign(__pi / 2.0f, z_imag));
  if (__spirv_ocl_fabs(z_real) == 1.0f && z_imag == 0.0f)
    return __SYCL_CMPLXF(
        __spirv_ocl_copysign(std::numeric_limits<float>::infinity(), z_real),
        __spirv_ocl_copysign(0.0f, z_imag));
  float __complex__ t1 = 1.0f + z;
  float __complex__ t2 = 1.0f - z;
  float __complex__ t3 =
      __divsc3(crealf(t1), cimagf(t1), crealf(t2), cimagf(t2));
  float __complex__ w = clogf(t3) / 2.0f;
  return __SYCL_CMPLXF(__spirv_ocl_copysign(crealf(w), z_real),
                       __spirv_ocl_copysign(cimagf(w), z_imag));
}
__SYCL_DEVICE_C
double __complex__ catanh(double __complex__ z) {
  double z_real = creal(z);
  double z_imag = cimag(z);
  const double __pi(__spirv_ocl_atan2(+0.0, -0.0));
  if (__spirv_IsInf(z_imag))
    return __SYCL_CMPLX(__spirv_ocl_copysign(0.0, z_real),
                        __spirv_ocl_copysign(__pi / 2.0, z_imag));
  if (__spirv_IsNan(z_imag)) {
    if (__spirv_IsInf(z_real) || z_real == 0)
      return __SYCL_CMPLX(__spirv_ocl_copysign(0.0, z_real), z_imag);
    return __SYCL_CMPLX(z_imag, z_imag);
  }
  if (__spirv_IsNan(z_real))
    return __SYCL_CMPLX(z_real, z_real);
  if (__spirv_IsInf(z_real))
    return __SYCL_CMPLX(__spirv_ocl_copysign(0.0, z_real),
                        __spirv_ocl_copysign(__pi / 2.0, z_imag));
  if (__spirv_ocl_fabs(z_real) == 1.0 && z_imag == 0.0)
    return __SYCL_CMPLX(
        __spirv_ocl_copysign(std::numeric_limits<double>::infinity(), z_real),
        __spirv_ocl_copysign(0.0, z_imag));
  double __complex__ t1 = 1.0 + z;
  double __complex__ t2 = 1.0 - z;
  double __complex__ t3 = __divdc3(creal(t1), cimag(t1), creal(t2), cimag(t2));
  double __complex__ w = clog(t3) / 2.0;
  return __SYCL_CMPLX(__spirv_ocl_copysign(creal(w), z_real),
                      __spirv_ocl_copysign(cimag(w), z_imag));
}

__SYCL_DEVICE_C
float __complex__ catanf(float __complex__ z) {
  float __complex__ w = catanhf(__SYCL_CMPLXF(-cimagf(z), crealf(z)));
  return __SYCL_CMPLXF(cimagf(w), -crealf(w));
}
__SYCL_DEVICE_C
double __complex__ catan(double __complex__ z) {
  double __complex__ w = catanh(__SYCL_CMPLX(-cimag(z), creal(z)));
  return __SYCL_CMPLX(cimag(w), -creal(w));
}

#undef __SYCL_CMPLXF
#undef __SYCL_DEVICE_C
#undef __SYCL_DEVICE
#endif // __SYCL_DEVICE_ONLY__
#endif // __SYCL_COMPLEX_IMPL_HPP__
