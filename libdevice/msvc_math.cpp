//==-- msvc_math.cpp - some msvc specific math functions for SYCL device ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__SPIR__) || defined(__SPIRV__) || defined(__NVPTX__)

#include "device.h"
#include <math.h>
union _Fval { // pun floating type as integer array
  unsigned short _Sh[8];
  float _Val;
};

union _Dconst {            // pun float types as integer array
  unsigned short _Word[2]; // TRANSITION, ABI: Twice as large as necessary.
  float _Float;
};

#define _F0 1 // little-endian
#define _F1 0

// IEEE 754 float properties
#define FHUGE_EXP (int)(_FMAX * 900L / 1000)

#define NBITS (16 + _FOFF)
#define FSIGN(x) (((_Fval *)(char *)&(x))->_Sh[_F0] & _FSIGN)

#define INIT(w0)                                                               \
  { 0, w0 }

#define _FXbig (float)((NBITS + 1) * 347L / 1000)

DEVICE_EXTERN_C_INLINE
float __devicelib_hypotf(float, float);

DEVICE_EXTERN_C_INLINE
float _hypotf(float x, float y) { return __devicelib_hypotf(x, y); }

DEVICE_EXTERN_C_INLINE
short _FDtest(float *px) { // categorize *px
  _Fval *ps = (_Fval *)(char *)px;
  short ret = 0;
  if ((ps->_Sh[_F0] & _FMASK) == _FMAX << _FOFF)
    ret = (short)((ps->_Sh[_F0] & _FFRAC) != 0 || ps->_Sh[_F1] != 0 ? _NANCODE
                                                                    : _INFCODE);
  else if ((ps->_Sh[_F0] & ~_FSIGN) != 0 || ps->_Sh[_F1] != 0)
    ret = (ps->_Sh[_F0] & _FMASK) == 0 ? _DENORM : _FINITE;

  return ret;
}

// Returns _FP_LT, _FP_GT or _FP_EQ based on the ordering
// relationship between x and y. '0' means unordered.
DEVICE_EXTERN_C_INLINE
int _fdpcomp(float x, float y) {
  int res = 0;
  if (_FDtest(&x) == _NANCODE || _FDtest(&y) == _NANCODE) {
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
int _fdsign(float x) { return FSIGN(x); }

// fpclassify() equivalent with a pointer argument.
DEVICE_EXTERN_C_INLINE
short _fdtest(float *px) {
  switch (_FDtest(px)) {
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
short _FDnorm(_Fval *ps) { // normalize float fraction
  short xchar;
  unsigned short sign = (unsigned short)(ps->_Sh[_F0] & _FSIGN);

  xchar = 1;
  if ((ps->_Sh[_F0] &= _FFRAC) != 0 || ps->_Sh[_F1]) { // nonzero, scale
    if (ps->_Sh[_F0] == 0) {
      ps->_Sh[_F0] = ps->_Sh[_F1];
      ps->_Sh[_F1] = 0;
      xchar -= 16;
    }

    for (; ps->_Sh[_F0] < 1 << _FOFF; --xchar) { // shift left by 1
      ps->_Sh[_F0] = (unsigned short)(ps->_Sh[_F0] << 1 | ps->_Sh[_F1] >> 15);
      ps->_Sh[_F1] <<= 1;
    }
    for (; 1 << (_FOFF + 1) <= ps->_Sh[_F0]; ++xchar) { // shift right by 1
      ps->_Sh[_F1] = (unsigned short)(ps->_Sh[_F1] >> 1 | ps->_Sh[_F0] << 15);
      ps->_Sh[_F0] >>= 1;
    }
    ps->_Sh[_F0] &= _FFRAC;
  }
  ps->_Sh[_F0] |= sign;
  return xchar;
}

DEVICE_EXTERN_C_INLINE
short _FDscale(float *px, long lexp) { // scale *px by 2^xexp with checking
  _Dconst _FInf = {INIT(_FMAX << _FOFF)};
  _Fval *ps = (_Fval *)(char *)px;
  short xchar = (short)((ps->_Sh[_F0] & _FMASK) >> _FOFF);

  if (xchar == _FMAX)
    return (short)((ps->_Sh[_F0] & _FFRAC) != 0 || ps->_Sh[_F1] != 0
                       ? _NANCODE
                       : _INFCODE);
  if (xchar == 0 && 0 < (xchar = _FDnorm(ps)))
    return 0;

  short ret = 0;
  if (0 < lexp && _FMAX - xchar <= lexp) { // overflow, return +/-INF
    *px = ps->_Sh[_F0] & _FSIGN ? -_FInf._Float : _FInf._Float;
    ret = _INFCODE;
  } else if (-xchar < lexp) { // finite result, repack
    ps->_Sh[_F0] =
        (unsigned short)(ps->_Sh[_F0] & ~_FMASK | (lexp + xchar) << _FOFF);
    ret = _FINITE;
  } else { // denormalized, scale
    unsigned short sign = (unsigned short)(ps->_Sh[_F0] & _FSIGN);

    ps->_Sh[_F0] = (unsigned short)(1 << _FOFF | ps->_Sh[_F0] & _FFRAC);
    lexp += xchar - 1;
    if (lexp < -(16 + 1 + _FOFF) || 0 <= lexp) { // underflow, return +/-0
      ps->_Sh[_F0] = sign;
      ps->_Sh[_F1] = 0;
      ret = 0;
    } else { // nonzero, align fraction
      ret = _FINITE;
      short xexp = (short)lexp;
      unsigned short psx = 0;

      if (xexp <= -16) { // scale by words
        psx = ps->_Sh[_F1] | (psx != 0 ? 1 : 0);
        ps->_Sh[_F1] = ps->_Sh[_F0];
        ps->_Sh[_F0] = 0;
        xexp += 16;
      }
      if ((xexp = (short)-xexp) != 0) { // scale by bits
        psx = (ps->_Sh[_F1] << (16 - xexp)) | (psx != 0 ? 1 : 0);
        ps->_Sh[_F1] = (unsigned short)(ps->_Sh[_F1] >> xexp |
                                        ps->_Sh[_F0] << (16 - xexp));
        ps->_Sh[_F0] >>= xexp;
      }

      ps->_Sh[_F0] |= sign;
      if ((0x8000 < psx || 0x8000 == psx && (ps->_Sh[_F1] & 0x0001) != 0) &&
          (++ps->_Sh[_F1] & 0xffff) == 0)
        ++ps->_Sh[_F0]; // round up
      else if (ps->_Sh[_F0] == sign && ps->_Sh[_F1] == 0)
        ret = 0;
    }
  }

  return ret;
}

DEVICE_EXTERN_C_INLINE
short _FExp(float *px, float y,
            short eoff) { // compute y * e^(*px), (*px) finite, |y| not huge
  static const float hugexp = FHUGE_EXP;
  _Dconst _FInf = {INIT(_FMAX << _FOFF)};
  static const float p[] = {1.0F, 60.09114349F};
  static const float q[] = {12.01517514F, 120.18228722F};
  static const float c1 = (22713.0F / 32768.0F);
  static const float c2 = 1.4286068203094172321214581765680755e-6F;
  static const float invln2 = 1.4426950408889634073599246810018921F;
  short ret = 0;
  if (*px < -hugexp || y == 0.0F) { // certain underflow
    *px = 0.0F;
  } else if (hugexp < *px) { // certain overflow
    *px = _FInf._Float;
    ret = _INFCODE;
  } else { // xexp won't overflow
    float g = *px * invln2;
    short xexp = (short)(g + (g < 0.0F ? -0.5F : +0.5F));

    g = xexp;
    g = (float)((*px - g * c1) - g * c2);
    if (-__FLT_EPSILON__ < g && g < __FLT_EPSILON__) {
      *px = y;
    } else { // g * g worth computing
      const float z = g * g;
      const float w = q[0] * z + q[1];

      g *= z + p[1];
      *px = (w + g) / (w - g) * 2.0F * y;
      --xexp;
    }
    ret = _FDscale(px, (long)xexp + eoff);
  }
  return ret;
}

DEVICE_EXTERN_C_INLINE
float _FCosh(float x, float y) { // compute y * cosh(x), |y| <= 1
  switch (_FDtest(&x)) {         // test for special codes
  case _NANCODE:
  case _INFCODE:
    return x;
  case 0:
    return y;
  default: // finite
    if (y == 0.0F)
      return y;

    if (x < 0.0F)
      x = -x;

    if (x < _FXbig) { // worth adding in exp(-x)
      _FExp(&x, 1.0F, -1);
      return y * (x + 0.25F / x);
    }
    _FExp(&x, y, -1);
    return x;
  }
}

DEVICE_EXTERN_C_INLINE
float _FSinh(float x, float y) { // compute y * sinh(x), |y| <= 1
  _Dconst _FRteps = {INIT((_FBIAS - NBITS / 2) << _FOFF)};
  static const float p[] = {0.00020400F, 0.00832983F, 0.16666737F, 0.99999998F};
  short neg;

  switch (_FDtest(&x)) { // test for special codes
  case _NANCODE:
    return x;
  case _INFCODE:
    return y != 0.0F ? x : FSIGN(x) ? -y : y;
  case 0:
    return x * y;
  default: // finite
    if (y == 0.0F)
      return x < 0.0F ? -y : y;

    if (x < 0.0F) {
      x = -x;
      neg = 1;
    } else
      neg = 0;

    if (x < _FRteps._Float) {
      x *= y; // x tiny
    } else if (x < 1.0F) {
      float w = x * x;

      x += ((p[0] * w + p[1]) * w + p[2]) * w * x;
      x *= y;
    } else if (x < _FXbig) { // worth adding in exp(-x)
      _FExp(&x, 1.0F, -1);
      x = y * (x - 0.25F / x);
    } else
      _FExp(&x, y, -1);

    return neg ? -x : x;
  }
}
#endif // __SPIR__ || __SPIRV__ || __NVPTX__
