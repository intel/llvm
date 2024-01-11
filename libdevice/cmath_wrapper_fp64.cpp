//==--- cmath_wrapper_fp64.cpp - wrappers for double precision C math library
// functions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device_math.h"

#if defined(__SPIR__) || defined(__NVPTX__)

// All exported functions in math and complex device libraries are weak
// reference. If users provide their own math or complex functions(with
// the prototype), functions in device libraries will be ignored and
// overrided by users' version.

DEVICE_EXTERN_C_INLINE
double fabs(double x) { return __devicelib_fabs(x); }

DEVICE_EXTERN_C_INLINE
double log(double x) { return __devicelib_log(x); }

DEVICE_EXTERN_C_INLINE
double round(double x) { return __devicelib_round(x); }

DEVICE_EXTERN_C_INLINE
double floor(double x) { return __devicelib_floor(x); }

DEVICE_EXTERN_C_INLINE
double exp(double x) { return __devicelib_exp(x); }

DEVICE_EXTERN_C_INLINE
double frexp(double x, int *exp) { return __devicelib_frexp(x, exp); }

DEVICE_EXTERN_C_INLINE
double ldexp(double x, int exp) { return __devicelib_ldexp(x, exp); }

DEVICE_EXTERN_C_INLINE
double log10(double x) { return __devicelib_log10(x); }

DEVICE_EXTERN_C_INLINE
double modf(double x, double *intpart) { return __devicelib_modf(x, intpart); }

DEVICE_EXTERN_C_INLINE
double exp2(double x) { return __devicelib_exp2(x); }

DEVICE_EXTERN_C_INLINE
double expm1(double x) { return __devicelib_expm1(x); }

DEVICE_EXTERN_C_INLINE
int ilogb(double x) { return __devicelib_ilogb(x); }

DEVICE_EXTERN_C_INLINE
double log1p(double x) { return __devicelib_log1p(x); }

DEVICE_EXTERN_C_INLINE
double log2(double x) { return __devicelib_log2(x); }

DEVICE_EXTERN_C_INLINE
double logb(double x) { return __devicelib_logb(x); }

DEVICE_EXTERN_C_INLINE
double sqrt(double x) { return __devicelib_sqrt(x); }

DEVICE_EXTERN_C_INLINE
double cbrt(double x) { return __devicelib_cbrt(x); }

DEVICE_EXTERN_C_INLINE
double hypot(double x, double y) { return __devicelib_hypot(x, y); }

DEVICE_EXTERN_C_INLINE
double erf(double x) { return __devicelib_erf(x); }

DEVICE_EXTERN_C_INLINE
double erfc(double x) { return __devicelib_erfc(x); }

DEVICE_EXTERN_C_INLINE
double tgamma(double x) { return __devicelib_tgamma(x); }

DEVICE_EXTERN_C_INLINE
double lgamma(double x) { return __devicelib_lgamma(x); }

DEVICE_EXTERN_C_INLINE
double fmod(double x, double y) { return __devicelib_fmod(x, y); }

DEVICE_EXTERN_C_INLINE
double remainder(double x, double y) { return __devicelib_remainder(x, y); }

DEVICE_EXTERN_C_INLINE
double remquo(double x, double y, int *q) {
  return __devicelib_remquo(x, y, q);
}

DEVICE_EXTERN_C_INLINE
double nextafter(double x, double y) { return __devicelib_nextafter(x, y); }

DEVICE_EXTERN_C_INLINE
double fdim(double x, double y) { return __devicelib_fdim(x, y); }

DEVICE_EXTERN_C_INLINE
double fma(double x, double y, double z) { return __devicelib_fma(x, y, z); }

DEVICE_EXTERN_C_INLINE
double sin(double x) { return __devicelib_sin(x); }

DEVICE_EXTERN_C_INLINE
double cos(double x) { return __devicelib_cos(x); }

DEVICE_EXTERN_C_INLINE
double tan(double x) { return __devicelib_tan(x); }

DEVICE_EXTERN_C_INLINE
double pow(double x, double y) { return __devicelib_pow(x, y); }

DEVICE_EXTERN_C_INLINE
double acos(double x) { return __devicelib_acos(x); }

DEVICE_EXTERN_C_INLINE
double asin(double x) { return __devicelib_asin(x); }

DEVICE_EXTERN_C_INLINE
double atan(double x) { return __devicelib_atan(x); }

DEVICE_EXTERN_C_INLINE
double atan2(double x, double y) { return __devicelib_atan2(x, y); }

DEVICE_EXTERN_C_INLINE
double cosh(double x) { return __devicelib_cosh(x); }

DEVICE_EXTERN_C_INLINE
double sinh(double x) { return __devicelib_sinh(x); }

DEVICE_EXTERN_C_INLINE
double tanh(double x) { return __devicelib_tanh(x); }

DEVICE_EXTERN_C_INLINE
double acosh(double x) { return __devicelib_acosh(x); }

DEVICE_EXTERN_C_INLINE
double asinh(double x) { return __devicelib_asinh(x); }

DEVICE_EXTERN_C_INLINE
double atanh(double x) { return __devicelib_atanh(x); }

DEVICE_EXTERN_C_INLINE
double scalbn(double x, int exp) { return __devicelib_scalbn(x, exp); }

#ifdef __NVPTX__
extern "C" SYCL_EXTERNAL double __nv_nearbyint(double);
DEVICE_EXTERN_C_INLINE
double nearbyint(double x) { return __nv_nearbyint(x); }

extern "C" SYCL_EXTERNAL double __nv_rint(double);
DEVICE_EXTERN_C_INLINE
double rint(double x) { return __nv_rint(x); }
#endif // __NVPTX__

#if defined(_MSC_VER)
#include <math.h>
// FLOAT PROPERTIES
#define _D0 3 // little-endian, small long doubles
#define _D1 2
#define _D2 1
#define _D3 0

// IEEE 754 double properties
#define HUGE_EXP (int)(_DMAX * 900L / 1000)

#define NBITS (48 + _DOFF)

#define INIT(w0)                                                               \
  { 0, 0, 0, w0 }

// double declarations
union _Dval { // pun floating type as integer array
  unsigned short _Sh[8];
  double _Val;
};

union _Dconst {            // pun float types as integer array
  unsigned short _Word[4]; // TRANSITION, ABI: Twice as large as necessary.
  double _Double;
};
#define DSIGN(x) (((_Dval *)(char *)&(x))->_Sh[_D0] & _DSIGN)

#define _Xbig (double)((NBITS + 1) * 347L / 1000)

DEVICE_EXTERN_C_INLINE
short _Dtest(double *px) { // categorize *px
  _Dval *ps = (_Dval *)(char *)px;

  short ret = 0;
  if ((ps->_Sh[_D0] & _DMASK) == _DMAX << _DOFF) {
    ret = (short)((ps->_Sh[_D0] & _DFRAC) != 0 || ps->_Sh[_D1] != 0 ||
                          ps->_Sh[_D2] != 0 || ps->_Sh[_D3] != 0
                      ? _NANCODE
                      : _INFCODE);
  } else if ((ps->_Sh[_D0] & ~_DSIGN) != 0 || ps->_Sh[_D1] != 0 ||
             ps->_Sh[_D2] != 0 || ps->_Sh[_D3] != 0)
    ret = (ps->_Sh[_D0] & _DMASK) == 0 ? _DENORM : _FINITE;

  return ret;
}

// Returns _FP_LT, _FP_GT or _FP_EQ based on the ordering
// relationship between x and y.
DEVICE_EXTERN_C_INLINE
int _dpcomp(double x, double y) {
  int res = 0;
  if (_Dtest(&x) == _NANCODE || _Dtest(&y) == _NANCODE) {
    // '0' means unordered.
    return res;
  }

  if (x < y)
    res |= _FP_LT;
  else if (x > y)
    res |= _FP_GT;
  else
    res |= _FP_EQ;

  return res;
}

// Returns 0, if the sign bit is not set, and non-zero otherwise.
DEVICE_EXTERN_C_INLINE
int _dsign(double x) { return DSIGN(x); }

// fpclassify() equivalent with a pointer argument.
DEVICE_EXTERN_C_INLINE
short _dtest(double *px) {
  switch (_Dtest(px)) {
  case _DENORM:
    return FP_SUBNORMAL;
  case _FINITE:
    return FP_NORMAL;
  case _INFCODE:
    return FP_INFINITE;
  case _NANCODE:
    return FP_NAN;
  }

  return FP_ZERO;
}

DEVICE_EXTERN_C_INLINE
short _Dnorm(_Dval *ps) { // normalize double fraction
  short xchar;
  unsigned short sign = (unsigned short)(ps->_Sh[_D0] & _DSIGN);

  xchar = 1;
  if ((ps->_Sh[_D0] &= _DFRAC) != 0 || ps->_Sh[_D1] || ps->_Sh[_D2] ||
      ps->_Sh[_D3]) {                        // nonzero, scale
    for (; ps->_Sh[_D0] == 0; xchar -= 16) { // shift left by 16
      ps->_Sh[_D0] = ps->_Sh[_D1];
      ps->_Sh[_D1] = ps->_Sh[_D2];
      ps->_Sh[_D2] = ps->_Sh[_D3];
      ps->_Sh[_D3] = 0;
    }
    for (; ps->_Sh[_D0] < 1 << _DOFF; --xchar) { // shift left by 1
      ps->_Sh[_D0] = (unsigned short)(ps->_Sh[_D0] << 1 | ps->_Sh[_D1] >> 15);
      ps->_Sh[_D1] = (unsigned short)(ps->_Sh[_D1] << 1 | ps->_Sh[_D2] >> 15);
      ps->_Sh[_D2] = (unsigned short)(ps->_Sh[_D2] << 1 | ps->_Sh[_D3] >> 15);
      ps->_Sh[_D3] <<= 1;
    }
    for (; 1 << (_DOFF + 1) <= ps->_Sh[_D0]; ++xchar) { // shift right by 1
      ps->_Sh[_D3] = (unsigned short)(ps->_Sh[_D3] >> 1 | ps->_Sh[_D2] << 15);
      ps->_Sh[_D2] = (unsigned short)(ps->_Sh[_D2] >> 1 | ps->_Sh[_D1] << 15);
      ps->_Sh[_D1] = (unsigned short)(ps->_Sh[_D1] >> 1 | ps->_Sh[_D0] << 15);
      ps->_Sh[_D0] >>= 1;
    }
    ps->_Sh[_D0] &= _DFRAC;
  }
  ps->_Sh[_D0] |= sign;
  return xchar;
}

DEVICE_EXTERN_C_INLINE
short _Dscale(double *px, long lexp) { // scale *px by 2^xexp with checking
  _Dval *ps = (_Dval *)(char *)px;
  _Dconst _Inf = {INIT(_DMAX << _DOFF)};
  short xchar = (short)((ps->_Sh[_D0] & _DMASK) >> _DOFF);

  if (xchar == _DMAX)
    return (short)((ps->_Sh[_D0] & _DFRAC) != 0 || ps->_Sh[_D1] != 0 ||
                           ps->_Sh[_D2] != 0 || ps->_Sh[_D3] != 0
                       ? _NANCODE
                       : _INFCODE);
  if (xchar == 0 && 0 < (xchar = _Dnorm(ps)))
    return 0;

  short ret = 0;
  if (0 < lexp && _DMAX - xchar <= lexp) { // overflow, return +/-INF
    *px = ps->_Sh[_D0] & _DSIGN ? -_Inf._Double : _Inf._Double;
    ret = _INFCODE;
  } else if (-xchar < lexp) { // finite result, repack
    ps->_Sh[_D0] =
        (unsigned short)(ps->_Sh[_D0] & ~_DMASK | (lexp + xchar) << _DOFF);
    ret = _FINITE;
  } else { // denormalized, scale
    unsigned short sign = (unsigned short)(ps->_Sh[_D0] & _DSIGN);

    ps->_Sh[_D0] = (unsigned short)(1 << _DOFF | ps->_Sh[_D0] & _DFRAC);
    lexp += xchar - 1;
    if (lexp < -(48 + 1 + _DOFF) ||
        0 <= lexp) { // certain underflow, return +/-0
      ps->_Sh[_D0] = sign;
      ps->_Sh[_D1] = 0;
      ps->_Sh[_D2] = 0;
      ps->_Sh[_D3] = 0;
      ret = 0;
    } else { // nonzero, align fraction
      short xexp = (short)lexp;
      unsigned short psx = 0;
      ret = _FINITE;

      for (; xexp <= -16; xexp += 16) { // scale by words
        psx = ps->_Sh[_D3] | (psx != 0 ? 1 : 0);
        ps->_Sh[_D3] = ps->_Sh[_D2];
        ps->_Sh[_D2] = ps->_Sh[_D1];
        ps->_Sh[_D1] = ps->_Sh[_D0];
        ps->_Sh[_D0] = 0;
      }
      if ((xexp = (short)-xexp) != 0) { // scale by bits
        psx = (ps->_Sh[_D3] << (16 - xexp)) | (psx != 0 ? 1 : 0);
        ps->_Sh[_D3] = (unsigned short)(ps->_Sh[_D3] >> xexp |
                                        ps->_Sh[_D2] << (16 - xexp));
        ps->_Sh[_D2] = (unsigned short)(ps->_Sh[_D2] >> xexp |
                                        ps->_Sh[_D1] << (16 - xexp));
        ps->_Sh[_D1] = (unsigned short)(ps->_Sh[_D1] >> xexp |
                                        ps->_Sh[_D0] << (16 - xexp));
        ps->_Sh[_D0] >>= xexp;
      }

      ps->_Sh[_D0] |= sign;
      if ((0x8000 < psx || 0x8000 == psx && (ps->_Sh[_D3] & 0x0001) != 0) &&
          (++ps->_Sh[_D3] & 0xffff) == 0 && (++ps->_Sh[_D2] & 0xffff) == 0 &&
          (++ps->_Sh[_D1] & 0xffff) == 0)
        ++ps->_Sh[_D0]; // round up
      else if (ps->_Sh[_D0] == sign && ps->_Sh[_D1] == 0 && ps->_Sh[_D2] == 0 &&
               ps->_Sh[_D3] == 0)
        ret = 0;
    }
  }
  return ret;
}

DEVICE_EXTERN_C_INLINE
short _Exp(double *px, double y,
           short eoff) { // compute y * e^(*px), (*px) finite, |y| not huge
  static const double invln2 = 1.4426950408889634073599246810018921;
  static const double c1 = 22713.0 / 32768.0;
  static const double c2 = 1.4286068203094172321214581765680755e-6;
  static const double p[] = {1.0, 420.30235984910635, 15132.70094680474802};
  static const double q[] = {30.01511290683317, 3362.72154416553028,
                             30265.40189360949691};

  _Dconst _Eps = {INIT((_DBIAS - NBITS - 1) << _DOFF)};
  _Dconst _Inf = {INIT(_DMAX << _DOFF)};
  short ret = 0;
  if (*px < -HUGE_EXP || y == 0.0) // certain underflow
    *px = 0.0;
  else if (HUGE_EXP < *px) { // certain overflow
    *px = _Inf._Double;
    ret = _INFCODE;
  } else { // xexp won't overflow
    double g = *px * invln2;
    short xexp = (short)(g + (g < 0.0 ? -0.5 : +0.5));
    g = xexp;
    g = (*px - g * c1) - g * c2;
    if (-_Eps._Double < g && g < _Eps._Double)
      *px = y;
    else { // g * g worth computing
      const double z = g * g;
      const double w = (q[0] * z + q[1]) * z + q[2];

      g *= (z + p[1]) * z + p[2];
      *px = (w + g) / (w - g) * 2.0 * y;
      --xexp;
    }
    ret = _Dscale(px, (long)xexp + eoff);
  }

  return ret;
}

DEVICE_EXTERN_C_INLINE
double _Cosh(double x, double y) { // compute y * cosh(x), |y| <= 1
  switch (_Dtest(&x)) {            // test for special codes
  case _NANCODE:
  case _INFCODE:
    return x;
  case 0:
    return y;
  default: // finite
    if (y == 0.0)
      return y;

    if (x < 0.0)
      x = -x;

    if (x < _Xbig) { // worth adding in exp(-x)
      _Exp(&x, 1.0, -1);
      return y * (x + 0.25 / x);
    }
    _Exp(&x, y, -1);
    return x;
  }
}

DEVICE_EXTERN_C_INLINE
double _Poly(double x, const double *tab, int n) { // compute polynomial
  double y;

  for (y = *tab; 0 <= --n;)
    y = y * x + *++tab;

  return y;
}

DEVICE_EXTERN_C_INLINE
double _Sinh(double x, double y) { // compute y * sinh(x), |y| <= 1

  short neg;
  // coefficients
  static const double p[] = {0.0000000001632881, 0.0000000250483893,
                             0.0000027557344615, 0.0001984126975233,
                             0.0083333333334816, 0.1666666666666574,
                             1.0000000000000001};
  _Dconst _Rteps = {INIT((_DBIAS - NBITS / 2) << _DOFF)};
  switch (_Dtest(&x)) { // test for special codes
  case _NANCODE:
    return x;
  case _INFCODE:
    return y != 0.0 ? x : DSIGN(x) ? -y : y;
  case 0:
    return x * y;
  default: // finite
    if (y == 0.0)
      return x < 0.0 ? -y : y;

    if (x < 0.0) {
      x = -x;
      neg = 1;
    } else
      neg = 0;

    if (x < _Rteps._Double)
      x *= y; // x tiny
    else if (x < 1.0) {
      double w = x * x;

      x += x * w * _Poly(w, p, 5);
      x *= y;
    } else if (x < _Xbig) { // worth adding in exp(-x)
      _Exp(&x, 1.0, -1);
      x = y * (x - 0.25 / x);
    } else
      _Exp(&x, y, -1);

    return neg ? -x : x;
  }
}
#endif // defined(_WIN32)
#endif // __SPIR__ || __NVPTX__
