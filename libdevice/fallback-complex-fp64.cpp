//==----- fallback-complex-fp64.cpp - double precision complex math functions
// for SPIR-V device --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_complex.h"

#if defined(__SPIR__) || defined(__SPIRV__)
#include <cmath>

// To support fallback device libraries on-demand loading, please update the
// DeviceLibFuncMap in llvm/tools/sycl-post-link/sycl-post-link.cpp if you add
// or remove any item in this file.
DEVICE_EXTERN_C_INLINE
double __devicelib_creal(double __complex__ z) { return __real__(z); }

DEVICE_EXTERN_C_INLINE
double __devicelib_cimag(double __complex__ z) { return __imag__(z); }

// __muldc3
// Returns: the product of a + ib and c + id
DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib___muldc3(double __a, double __b, double __c,
                                        double __d) {
  double __ac = __a * __c;
  double __bd = __b * __d;
  double __ad = __a * __d;
  double __bc = __b * __c;
  double __complex__ z;
  z = CMPLX((__ac - __bd), (__ad + __bc));
  if (__spirv_IsNan(__devicelib_creal(z)) &&
      __spirv_IsNan(__devicelib_cimag(z))) {
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
      z = CMPLX((INFINITY * (__a * __c - __b * __d)),
                (INFINITY * (__a * __d + __b * __c)));
    }
  }
  return z;
}

// __divdc3
// Returns: the quotient of (a + ib) / (c + id)
DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib___divdc3(double __a, double __b, double __c,
                                        double __d) {
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
  z = CMPLX(z_real, z_imag);
  if (__spirv_IsNan(z_real) && __spirv_IsNan(z_imag)) {
    if ((__denom == 0.0) && (!__spirv_IsNan(__a) || !__spirv_IsNan(__b))) {
      z_real = __spirv_ocl_copysign((double)INFINITY, __c) * __a;
      z_imag = __spirv_ocl_copysign((double)INFINITY, __c) * __b;
      z = CMPLX(z_real, z_imag);
    } else if ((__spirv_IsInf(__a) || __spirv_IsInf(__b)) &&
               __spirv_IsFinite(__c) && __spirv_IsFinite(__d)) {
      __a = __spirv_ocl_copysign(__spirv_IsInf(__a) ? 1.0 : 0.0, __a);
      __b = __spirv_ocl_copysign(__spirv_IsInf(__b) ? 1.0 : 0.0, __b);
      z_real = INFINITY * (__a * __c + __b * __d);
      z_imag = INFINITY * (__b * __c - __a * __d);
      z = CMPLX(z_real, z_imag);
    } else if (__spirv_IsInf(__logbw) && __logbw > 0.0 &&
               __spirv_IsFinite(__a) && __spirv_IsFinite(__b)) {
      __c = __spirv_ocl_copysign(__spirv_IsInf(__c) ? 1.0 : 0.0, __c);
      __d = __spirv_ocl_copysign(__spirv_IsInf(__d) ? 1.0 : 0.0, __d);
      z_real = 0.0 * (__a * __c + __b * __d);
      z_imag = 0.0 * (__b * __c - __a * __d);
      z = CMPLX(z_real, z_imag);
    }
  }
  return z;
}

DEVICE_EXTERN_C_INLINE
double __devicelib_cabs(double __complex__ z) {
  return __spirv_ocl_hypot(__devicelib_creal(z), __devicelib_cimag(z));
}

DEVICE_EXTERN_C_INLINE
double __devicelib_carg(double __complex__ z) {
  return __spirv_ocl_atan2(__devicelib_cimag(z), __devicelib_creal(z));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_cproj(double __complex__ z) {
  double __complex__ r = z;
  if (__spirv_IsInf(__devicelib_creal(z)) ||
      __spirv_IsInf(__devicelib_cimag(z)))
    r = CMPLX(INFINITY, __spirv_ocl_copysign(0.0, __devicelib_cimag(z)));
  return r;
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_cexp(double __complex__ z) {
  double z_imag = __devicelib_cimag(z);
  double z_real = __devicelib_creal(z);
  if (__spirv_IsInf(z_real)) {
    if (z_real < 0.0) {
      if (!__spirv_IsFinite(z_imag))
        z_imag = 1.0;
    } else if (z_imag == 0.0 || !__spirv_IsFinite(z_imag)) {
      if (__spirv_IsInf(z_imag))
        z_imag = NAN;
      return CMPLX(z_real, z_imag);
    }
  } else if (__spirv_IsNan(z_real) && (z_imag == 0.0)) {
    return z;
  }
  double __e = __spirv_ocl_exp(z_real);
  return CMPLX((__e * __spirv_ocl_cos(z_imag)),
               (__e * __spirv_ocl_sin(z_imag)));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_clog(double __complex__ z) {
  return CMPLX(__spirv_ocl_log(__devicelib_cabs(z)), __devicelib_carg(z));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_cpow(double __complex__ x,
                                    double __complex__ y) {
  double __complex__ t = __devicelib_clog(x);
  double __complex__ w =
      __devicelib___muldc3(__devicelib_creal(y), __devicelib_cimag(y),
                           __devicelib_creal(t), __devicelib_cimag(t));
  return __devicelib_cexp(w);
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_cpolar(double rho, double theta) {
  if (__spirv_IsNan(rho) || __spirv_SignBitSet(rho))
    return CMPLX(NAN, NAN);
  if (__spirv_IsNan(theta)) {
    if (__spirv_IsInf(rho))
      return CMPLX(rho, theta);
    return CMPLX(theta, theta);
  }
  if (__spirv_IsInf(theta)) {
    if (__spirv_IsInf(rho))
      return CMPLX(rho, NAN);
    return CMPLX(NAN, NAN);
  }
  double x = rho * __spirv_ocl_cos(theta);
  if (__spirv_IsNan(x))
    x = 0;
  double y = rho * __spirv_ocl_sin(theta);
  if (__spirv_IsNan(y))
    y = 0;
  return CMPLX(x, y);
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_csqrt(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  if (__spirv_IsInf(z_imag))
    return CMPLX(INFINITY, z_imag);
  if (__spirv_IsInf(z_real)) {
    if (z_real > 0.0)
      return CMPLX(z_real, __spirv_IsNan(z_imag)
                               ? z_imag
                               : __spirv_ocl_copysign(0.0, z_imag));
    return CMPLX(__spirv_IsNan(z_imag) ? z_imag : 0.0,
                 __spirv_ocl_copysign(z_real, z_imag));
  }
  return __devicelib_cpolar(__spirv_ocl_sqrt(__devicelib_cabs(z)),
                            __devicelib_carg(z) / 2.0);
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_csinh(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  if (__spirv_IsInf(z_real) && !__spirv_IsFinite(z_imag))
    return CMPLX(z_real, NAN);
  if (z_real == 0 && !__spirv_IsFinite(z_imag))
    return CMPLX(z_real, NAN);
  if (z_imag == 0 && !__spirv_IsFinite(z_real))
    return z;
  return CMPLX(__spirv_ocl_sinh(z_real) * __spirv_ocl_cos(z_imag),
               __spirv_ocl_cosh(z_real) * __spirv_ocl_sin(z_imag));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_ccosh(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  if (__spirv_IsInf(z_real) && !__spirv_IsFinite(z_imag))
    return CMPLX(__spirv_ocl_fabs(z_real), NAN);
  if (z_real == 0 && !__spirv_IsFinite(z_imag))
    return CMPLX(NAN, z_real);
  if (z_real == 0 && z_imag == 0)
    return CMPLX(1.0f, z_imag);
  if (z_imag == 0 && !__spirv_IsFinite(z_real))
    return CMPLX(__spirv_ocl_fabs(z_real), z_imag);
  return CMPLX(__spirv_ocl_cosh(z_real) * __spirv_ocl_cos(z_imag),
               __spirv_ocl_sinh(z_real) * __spirv_ocl_sin(z_imag));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_ctanh(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  if (__spirv_IsInf(z_real)) {
    if (!__spirv_IsFinite(z_imag))
      return CMPLX(1.0, 0.0);
    return CMPLX(1.0, __spirv_ocl_copysign(0.0, __spirv_ocl_sin(2.0 * z_imag)));
  }
  if (__spirv_IsNan(z_real) && z_imag == 0)
    return z;
  double __2r(2.0 * z_real);
  double __2i(2.0 * z_imag);
  double __d(__spirv_ocl_cosh(__2r) + __spirv_ocl_cos(__2i));
  double __2rsh(__spirv_ocl_sinh(__2r));
  if (__spirv_IsInf(__2rsh) && __spirv_IsInf(__d))
    return CMPLX(((__2rsh > 0.0) ? 1.0 : -1.0), ((__2i > 0.0) ? 0.0 : -0.0));
  return CMPLX(__2rsh / __d, __spirv_ocl_sin(__2i) / __d);
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_csin(double __complex__ z) {
  double __complex__ w =
      __devicelib_csinh(CMPLX(-__devicelib_cimag(z), __devicelib_creal(z)));
  return CMPLX(__devicelib_cimag(w), -__devicelib_creal(w));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_ccos(double __complex__ z) {
  return __devicelib_ccosh(CMPLX(-__devicelib_cimag(z), __devicelib_creal(z)));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_ctan(double __complex__ z) {
  double __complex__ w =
      __devicelib_ctanh(CMPLX(-__devicelib_cimag(z), __devicelib_creal(z)));
  return CMPLX(__devicelib_cimag(w), -__devicelib_creal(w));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __sqr(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  return CMPLX((z_real + z_imag) * (z_real - z_imag), 2.0 * z_real * z_imag);
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_cacos(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  const double __pi(__spirv_ocl_atan2(+0.0, -0.0));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return CMPLX(z_imag, z_real);
    if (__spirv_IsInf(z_imag)) {
      if (z_real < 0.0)
        return CMPLX(0.75 * __pi, -z_imag);
      return CMPLX(0.25 * __pi, -z_imag);
    }
    if (z_real < 0.0)
      return CMPLX(__pi, __spirv_SignBitSet(z_imag) ? -z_real : z_real);
    return CMPLX(0.0f, __spirv_SignBitSet(z_imag) ? z_real : -z_real);
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return CMPLX(z_real, -z_imag);
    return CMPLX(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return CMPLX(__pi / 2.0, -z_imag);
  if (z_real == 0 && (z_imag == 0 || __spirv_IsNan(z_imag)))
    return CMPLX(__pi / 2.0, -z_imag);
  double __complex__ w =
      __devicelib_clog(z + __devicelib_csqrt(__sqr(z) - 1.0));
  if (__spirv_SignBitSet(z_imag))
    return CMPLX(__spirv_ocl_fabs(__devicelib_cimag(w)),
                 __spirv_ocl_fabs(__devicelib_creal(w)));
  return CMPLX(__spirv_ocl_fabs(__devicelib_cimag(w)),
               -__spirv_ocl_fabs(__devicelib_creal(w)));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_casinh(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  const double __pi(__spirv_ocl_atan2(+0.0, -0.0));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return z;
    if (__spirv_IsInf(z_imag))
      return CMPLX(z_real, __spirv_ocl_copysign(__pi * 0.25, z_imag));
    return CMPLX(z_real, __spirv_ocl_copysign(0.0, z_imag));
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return CMPLX(z_imag, z_real);
    if (z_imag == 0)
      return z;
    return CMPLX(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return CMPLX(__spirv_ocl_copysign(z_imag, z_real),
                 __spirv_ocl_copysign(__pi / 2.0, z_imag));
  double __complex__ w =
      __devicelib_clog(z + __devicelib_csqrt(__sqr(z) + 1.0));
  return CMPLX(__spirv_ocl_copysign(__devicelib_creal(w), z_real),
               __spirv_ocl_copysign(__devicelib_cimag(w), z_imag));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_casin(double __complex__ z) {
  double __complex__ w =
      __devicelib_casinh(CMPLX(-__devicelib_cimag(z), __devicelib_creal(z)));
  return CMPLX(__devicelib_cimag(w), -__devicelib_creal(w));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_cacosh(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  const double __pi(__spirv_ocl_atan2(+0.0, -0.0));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return CMPLX(__spirv_ocl_fabs(z_real), z_imag);
    if (__spirv_IsInf(z_imag)) {
      if (z_real > 0)
        return CMPLX(z_real, __spirv_ocl_copysign(__pi * 0.25f, z_imag));
      else
        return CMPLX(-z_real, __spirv_ocl_copysign(__pi * 0.75f, z_imag));
    }
    if (z_real < 0)
      return CMPLX(-z_real, __spirv_ocl_copysign(__pi, z_imag));
    return CMPLX(z_real, __spirv_ocl_copysign(0.0, z_imag));
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return CMPLX(__spirv_ocl_fabs(z_imag), z_real);
    return CMPLX(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return CMPLX(__spirv_ocl_fabs(z_imag),
                 __spirv_ocl_copysign(__pi / 2.0, z_imag));
  double __complex__ w =
      __devicelib_clog(z + __devicelib_csqrt(__sqr(z) - 1.0));
  return CMPLX(__spirv_ocl_copysign(__devicelib_creal(w), 0.0),
               __spirv_ocl_copysign(__devicelib_cimag(w), z_imag));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_catanh(double __complex__ z) {
  double z_real = __devicelib_creal(z);
  double z_imag = __devicelib_cimag(z);
  const double __pi(__spirv_ocl_atan2(+0.0, -0.0));
  if (__spirv_IsInf(z_imag))
    return CMPLX(__spirv_ocl_copysign(0.0, z_real),
                 __spirv_ocl_copysign(__pi / 2.0, z_imag));
  if (__spirv_IsNan(z_imag)) {
    if (__spirv_IsInf(z_real) || z_real == 0)
      return CMPLX(__spirv_ocl_copysign(0.0, z_real), z_imag);
    return CMPLX(z_imag, z_imag);
  }
  if (__spirv_IsNan(z_real))
    return CMPLX(z_real, z_real);
  if (__spirv_IsInf(z_real))
    return CMPLX(__spirv_ocl_copysign(0.0, z_real),
                 __spirv_ocl_copysign(__pi / 2.0, z_imag));
  if (__spirv_ocl_fabs(z_real) == 1.0 && z_imag == 0.0)
    return CMPLX(__spirv_ocl_copysign(static_cast<double>(INFINITY), z_real),
                 __spirv_ocl_copysign(0.0, z_imag));
  double __complex__ t1 = 1.0 + z;
  double __complex__ t2 = 1.0 - z;
  double __complex__ t3 =
      __devicelib___divdc3(__devicelib_creal(t1), __devicelib_cimag(t1),
                           __devicelib_creal(t2), __devicelib_cimag(t2));
  double __complex__ w = __devicelib_clog(t3) / 2.0;
  return CMPLX(__spirv_ocl_copysign(__devicelib_creal(w), z_real),
               __spirv_ocl_copysign(__devicelib_cimag(w), z_imag));
}

DEVICE_EXTERN_C_INLINE
double __complex__ __devicelib_catan(double __complex__ z) {
  double __complex__ w =
      __devicelib_catanh(CMPLX(-__devicelib_cimag(z), __devicelib_creal(z)));
  return CMPLX(__devicelib_cimag(w), -__devicelib_creal(w));
}
#endif // __SPIR__ || __SPIRV__
