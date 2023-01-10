//==----- fallback-complex.cpp - complex math functions for SPIR-V device --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_complex.h"

#ifdef __SPIR__
#include <cmath>

// To support fallback device libraries on-demand loading, please update the
// DeviceLibFuncMap in llvm/tools/sycl-post-link/sycl-post-link.cpp if you add
// or remove any item in this file.
DEVICE_EXTERN_C_INLINE
float __devicelib_crealf(float __complex__ z) { return __real__(z); }

DEVICE_EXTERN_C_INLINE
float __devicelib_cimagf(float __complex__ z) { return __imag__(z); }

// __mulsc3
// Returns: the product of a + ib and c + id
DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib___mulsc3(float __a, float __b, float __c,
                                       float __d) {
  float __ac = __a * __c;
  float __bd = __b * __d;
  float __ad = __a * __d;
  float __bc = __b * __c;
  float __complex__ z;
  z = CMPLXF((__ac - __bd), (__ad + __bc));
  if (__spirv_IsNan(__devicelib_crealf(z)) &&
      __spirv_IsNan(__devicelib_cimagf(z))) {
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
      z = CMPLXF((INFINITY * (__a * __c - __b * __d)),
                 (INFINITY * (__a * __d + __b * __c)));
    }
  }
  return z;
}

// __divsc3
// Returns: the quotient of (a + ib) / (c + id)
// FIXME: divsc3/divdc3 have overflow issue when dealing with large number.
// And this overflow issue is from libc++/compiler-rt's implementation.
DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib___divsc3(float __a, float __b, float __c,
                                       float __d) {
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
  z = CMPLXF(z_real, z_imag);
  if (__spirv_IsNan(z_real) && __spirv_IsNan(z_imag)) {
    if ((__denom == 0.0f) && (!__spirv_IsNan(__a) || !__spirv_IsNan(__b))) {
      z_real = __spirv_ocl_copysign(INFINITY, __c) * __a;
      z_imag = __spirv_ocl_copysign(INFINITY, __c) * __b;
      z = CMPLXF(z_real, z_imag);
    } else if ((__spirv_IsInf(__a) || __spirv_IsInf(__b)) &&
               __spirv_IsFinite(__c) && __spirv_IsFinite(__d)) {
      __a = __spirv_ocl_copysign(__spirv_IsInf(__a) ? 1.0f : 0.0f, __a);
      __b = __spirv_ocl_copysign(__spirv_IsInf(__b) ? 1.0f : 0.0f, __b);
      z_real = INFINITY * (__a * __c + __b * __d);
      z_imag = INFINITY * (__b * __c - __a * __d);
      z = CMPLXF(z_real, z_imag);
    } else if (__spirv_IsInf(__logbw) && __logbw > 0.0f &&
               __spirv_IsFinite(__a) && __spirv_IsFinite(__b)) {
      __c = __spirv_ocl_copysign(__spirv_IsInf(__c) ? 1.0f : 0.0f, __c);
      __d = __spirv_ocl_copysign(__spirv_IsInf(__d) ? 1.0f : 0.0f, __d);
      z_real = 0.0f * (__a * __c + __b * __d);
      z_imag = 0.0f * (__b * __c - __a * __d);
      z = CMPLXF(z_real, z_imag);
    }
  }
  return z;
}

DEVICE_EXTERN_C_INLINE
float __devicelib_cargf(float __complex__ z) {
  return __spirv_ocl_atan2(__devicelib_cimagf(z), __devicelib_crealf(z));
}

DEVICE_EXTERN_C_INLINE
float __devicelib_cabsf(float __complex__ z) {
  return __spirv_ocl_hypot(__devicelib_crealf(z), __devicelib_cimagf(z));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_cprojf(float __complex__ z) {
  float __complex__ r = z;
  if (__spirv_IsInf(__devicelib_crealf(z)) ||
      __spirv_IsInf(__devicelib_cimagf(z)))
    r = CMPLXF(INFINITY, __spirv_ocl_copysign(0.0f, __devicelib_cimagf(z)));
  return r;
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_cexpf(float __complex__ z) {
  float z_imag = __devicelib_cimagf(z);
  float z_real = __devicelib_crealf(z);
  if (__spirv_IsInf(z_real)) {
    if (z_real < 0.0f) {
      if (!__spirv_IsFinite(z_imag))
        z_imag = 1.0f;
    } else if (z_imag == 0.0f || !__spirv_IsFinite(z_imag)) {
      if (__spirv_IsInf(z_imag))
        z_imag = NAN;
      return CMPLXF(z_real, z_imag);
    }
  } else if (__spirv_IsNan(z_real) && (z_imag == 0.0f)) {
    return z;
  }
  float __e = __spirv_ocl_exp(z_real);
  return CMPLXF((__e * __spirv_ocl_cos(z_imag)),
                (__e * __spirv_ocl_sin(z_imag)));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_clogf(float __complex__ z) {
  return CMPLXF(__spirv_ocl_log(__devicelib_cabsf(z)), __devicelib_cargf(z));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_cpowf(float __complex__ x, float __complex__ y) {
  float __complex__ t = __devicelib_clogf(x);
  float __complex__ w =
      __devicelib___mulsc3(__devicelib_crealf(y), __devicelib_cimagf(y),
                           __devicelib_crealf(t), __devicelib_cimagf(t));
  return __devicelib_cexpf(w);
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_cpolarf(float rho, float theta) {
  if (__spirv_IsNan(rho) || __spirv_SignBitSet(rho))
    return CMPLXF(NAN, NAN);
  if (__spirv_IsNan(theta)) {
    if (__spirv_IsInf(rho))
      return CMPLXF(rho, theta);
    return CMPLXF(theta, theta);
  }
  if (__spirv_IsInf(theta)) {
    if (__spirv_IsInf(rho))
      return CMPLXF(rho, NAN);
    return CMPLXF(NAN, NAN);
  }
  float x = rho * __spirv_ocl_cos(theta);
  if (__spirv_IsNan(x))
    x = 0;
  float y = rho * __spirv_ocl_sin(theta);
  if (__spirv_IsNan(y))
    y = 0;
  return CMPLXF(x, y);
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_csqrtf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  if (__spirv_IsInf(z_imag))
    return CMPLXF(INFINITY, z_imag);
  if (__spirv_IsInf(z_real)) {
    if (z_real > 0.0f)
      return CMPLXF(z_real, __spirv_IsNan(z_imag)
                                ? z_imag
                                : __spirv_ocl_copysign(0.0f, z_imag));
    return CMPLXF(__spirv_IsNan(z_imag) ? z_imag : 0.0f,
                  __spirv_ocl_copysign(z_real, z_imag));
  }
  return __devicelib_cpolarf(__spirv_ocl_sqrt(__devicelib_cabsf(z)),
                             __devicelib_cargf(z) / 2.0f);
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_csinhf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  if (__spirv_IsInf(z_real) && !__spirv_IsFinite(z_imag))
    return CMPLXF(z_real, NAN);
  if (z_real == 0 && !__spirv_IsFinite(z_imag))
    return CMPLXF(z_real, NAN);
  if (z_imag == 0 && !__spirv_IsFinite(z_real))
    return z;
  return CMPLXF(__spirv_ocl_sinh(z_real) * __spirv_ocl_cos(z_imag),
                __spirv_ocl_cosh(z_real) * __spirv_ocl_sin(z_imag));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_ccoshf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  if (__spirv_IsInf(z_real) && !__spirv_IsFinite(z_imag))
    return CMPLXF(__spirv_ocl_fabs(z_real), NAN);
  if (z_real == 0 && !__spirv_IsFinite(z_imag))
    return CMPLXF(NAN, z_real);
  if (z_real == 0 && z_imag == 0)
    return CMPLXF(1.0f, z_imag);
  if (z_imag == 0 && !__spirv_IsFinite(z_real))
    return CMPLXF(__spirv_ocl_fabs(z_real), z_imag);
  return CMPLXF(__spirv_ocl_cosh(z_real) * __spirv_ocl_cos(z_imag),
                __spirv_ocl_sinh(z_real) * __spirv_ocl_sin(z_imag));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_ctanhf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  if (__spirv_IsInf(z_real)) {
    if (!__spirv_IsFinite(z_imag))
      return CMPLXF(1.0f, 0.0f);
    return CMPLXF(1.0f,
                  __spirv_ocl_copysign(0.0f, __spirv_ocl_sin(2.0f * z_imag)));
  }
  if (__spirv_IsNan(z_real) && z_imag == 0)
    return z;
  float __2r(2.0f * z_real);
  float __2i(2.0f * z_imag);
  float __d(__spirv_ocl_cosh(__2r) + __spirv_ocl_cos(__2i));
  float __2rsh(__spirv_ocl_sinh(__2r));
  if (__spirv_IsInf(__2rsh) && __spirv_IsInf(__d))
    return CMPLXF(((__2rsh > 0.0f) ? 1.0f : -1.0f),
                  ((__2i > 0.0f) ? 0.0f : -0.0f));
  return CMPLXF(__2rsh / __d, __spirv_ocl_sin(__2i) / __d);
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_csinf(float __complex__ z) {
  float __complex__ w =
      __devicelib_csinhf(CMPLXF(-__devicelib_cimagf(z), __devicelib_crealf(z)));
  return CMPLXF(__devicelib_cimagf(w), -__devicelib_crealf(w));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_ccosf(float __complex__ z) {
  return __devicelib_ccoshf(
      CMPLXF(-__devicelib_cimagf(z), __devicelib_crealf(z)));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_ctanf(float __complex__ z) {
  float __complex__ w =
      __devicelib_ctanhf(CMPLXF(-__devicelib_cimagf(z), __devicelib_crealf(z)));
  return CMPLXF(__devicelib_cimagf(w), -__devicelib_crealf(w));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __sqrf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  return CMPLXF((z_real + z_imag) * (z_real - z_imag), 2.0f * z_real * z_imag);
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_cacosf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  const float __pi(__spirv_ocl_atan2(+0.0f, -0.0f));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return CMPLXF(z_imag, z_real);
    if (__spirv_IsInf(z_imag)) {
      if (z_real < 0.0f)
        return CMPLXF(0.75f * __pi, -z_imag);
      return CMPLXF(0.25f * __pi, -z_imag);
    }
    if (z_real < 0.0f)
      return CMPLXF(__pi, __spirv_SignBitSet(z_imag) ? -z_real : z_real);
    return CMPLXF(0.0f, __spirv_SignBitSet(z_imag) ? z_real : -z_real);
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return CMPLXF(z_real, -z_imag);
    return CMPLXF(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return CMPLXF(__pi / 2.0f, -z_imag);
  if (z_real == 0 && (z_imag == 0 || __spirv_IsNan(z_imag)))
    return CMPLXF(__pi / 2.0f, -z_imag);
  float __complex__ w =
      __devicelib_clogf(z + __devicelib_csqrtf(__sqrf(z) - 1.0f));
  if (__spirv_SignBitSet(z_imag))
    return CMPLXF(__spirv_ocl_fabs(__devicelib_cimagf(w)),
                  __spirv_ocl_fabs(__devicelib_crealf(w)));
  return CMPLXF(__spirv_ocl_fabs(__devicelib_cimagf(w)),
                -__spirv_ocl_fabs(__devicelib_crealf(w)));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_casinhf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  const float __pi(__spirv_ocl_atan2(+0.0f, -0.0f));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return z;
    if (__spirv_IsInf(z_imag))
      return CMPLXF(z_real, __spirv_ocl_copysign(__pi * 0.25f, z_imag));
    return CMPLXF(z_real, __spirv_ocl_copysign(0.0f, z_imag));
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return CMPLXF(z_imag, z_real);
    if (z_imag == 0)
      return z;
    return CMPLXF(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return CMPLXF(__spirv_ocl_copysign(z_imag, z_real),
                  __spirv_ocl_copysign(__pi / 2.0f, z_imag));
  float __complex__ w =
      __devicelib_clogf(z + __devicelib_csqrtf(__sqrf(z) + 1.0f));
  return CMPLXF(__spirv_ocl_copysign(__devicelib_crealf(w), z_real),
                __spirv_ocl_copysign(__devicelib_cimagf(w), z_imag));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_casinf(float __complex__ z) {
  float __complex__ w = __devicelib_casinhf(
      CMPLXF(-__devicelib_cimagf(z), __devicelib_crealf(z)));
  return CMPLXF(__devicelib_cimagf(w), -__devicelib_crealf(w));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_cacoshf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  const float __pi(__spirv_ocl_atan2(+0.0f, -0.0f));
  if (__spirv_IsInf(z_real)) {
    if (__spirv_IsNan(z_imag))
      return CMPLXF(__spirv_ocl_fabs(z_real), z_imag);
    if (__spirv_IsInf(z_imag)) {
      if (z_real > 0)
        return CMPLXF(z_real, __spirv_ocl_copysign(__pi * 0.25f, z_imag));
      else
        return CMPLXF(-z_real, __spirv_ocl_copysign(__pi * 0.75f, z_imag));
    }
    if (z_real < 0)
      return CMPLXF(-z_real, __spirv_ocl_copysign(__pi, z_imag));
    return CMPLXF(z_real, __spirv_ocl_copysign(0.0f, z_imag));
  }
  if (__spirv_IsNan(z_real)) {
    if (__spirv_IsInf(z_imag))
      return CMPLXF(__spirv_ocl_fabs(z_imag), z_real);
    return CMPLXF(z_real, z_real);
  }
  if (__spirv_IsInf(z_imag))
    return CMPLXF(__spirv_ocl_fabs(z_imag),
                  __spirv_ocl_copysign(__pi / 2.0f, z_imag));
  float __complex__ w =
      __devicelib_clogf(z + __devicelib_csqrtf(__sqrf(z) - 1.0f));
  return CMPLXF(__spirv_ocl_copysign(__devicelib_crealf(w), 0.0f),
                __spirv_ocl_copysign(__devicelib_cimagf(w), z_imag));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_catanhf(float __complex__ z) {
  float z_real = __devicelib_crealf(z);
  float z_imag = __devicelib_cimagf(z);
  const float __pi(__spirv_ocl_atan2(+0.0f, -0.0f));
  if (__spirv_IsInf(z_imag))
    return CMPLXF(__spirv_ocl_copysign(0.0f, z_real),
                  __spirv_ocl_copysign(__pi / 2.0f, z_imag));
  if (__spirv_IsNan(z_imag)) {
    if (__spirv_IsInf(z_real) || z_real == 0)
      return CMPLXF(__spirv_ocl_copysign(0.0f, z_real), z_imag);
    return CMPLXF(z_imag, z_imag);
  }
  if (__spirv_IsNan(z_real))
    return CMPLXF(z_real, z_real);
  if (__spirv_IsInf(z_real))
    return CMPLXF(__spirv_ocl_copysign(0.0f, z_real),
                  __spirv_ocl_copysign(__pi / 2.0f, z_imag));
  if (__spirv_ocl_fabs(z_real) == 1.0f && z_imag == 0.0f)
    return CMPLXF(__spirv_ocl_copysign(INFINITY, z_real),
                  __spirv_ocl_copysign(0.0f, z_imag));
  float __complex__ t1 = 1.0f + z;
  float __complex__ t2 = 1.0f - z;
  float __complex__ t3 =
      __devicelib___divsc3(__devicelib_crealf(t1), __devicelib_cimagf(t1),
                           __devicelib_crealf(t2), __devicelib_cimagf(t2));
  float __complex__ w = __devicelib_clogf(t3) / 2.0f;
  return CMPLXF(__spirv_ocl_copysign(__devicelib_crealf(w), z_real),
                __spirv_ocl_copysign(__devicelib_cimagf(w), z_imag));
}

DEVICE_EXTERN_C_INLINE
float __complex__ __devicelib_catanf(float __complex__ z) {
  float __complex__ w = __devicelib_catanhf(
      CMPLXF(-__devicelib_cimagf(z), __devicelib_crealf(z)));
  return CMPLXF(__devicelib_cimagf(w), -__devicelib_crealf(w));
}
#endif // __SPIR__
