//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_mad.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>
#include <clc/opencl/clc.h>
#include <libspirv/spirv.h>

/*
 compute pow using log and exp
 x^y = exp(y * log(x))

 we take care not to lose precision in the intermediate steps

 When computing log, calculate it in splits,

 r = f * (p_invead + p_inv_tail)
 r = rh + rt

 calculate log polynomial using r, in end addition, do
 poly = poly + ((rh-r) + rt)

 lth = -r
 ltt = ((xexp * log2_t) - poly) + logT
 lt = lth + ltt

 lh = (xexp * log2_h) + logH
 l = lh + lt

 Calculate final log answer as gh and gt,
 gh = l & higher-half bits
 gt = (((ltt - (lt - lth)) + ((lh - l) + lt)) + (l - gh))

 yh = y & higher-half bits
 yt = y - yh

 Before entering computation of exp,
 vs = ((yt*gt + yt*gh) + yh*gt)
 v = vs + yh*gh
 vt = ((yh*gh - v) + vs)

 In calculation of exp, add vt to r that is used for poly
 At the end of exp, do
 ((((expT * poly) + expT) + expH*poly) + expH)
*/

_CLC_DEF _CLC_OVERLOAD float __clc_pow(float x, float y) {

  int ix = __clc_as_int(x);
  int ax = ix & EXSIGNBIT_SP32;
  int xpos = ix == ax;

  int iy = __clc_as_int(y);
  int ay = iy & EXSIGNBIT_SP32;
  int ypos = iy == ay;

  /* Extra precise log calculation
   *  First handle case that x is close to 1
   */
  float r = 1.0f - __clc_as_float(ax);
  int near1 = __spirv_ocl_fabs(r) < 0x1.0p-4f;
  float r2 = r * r;

  /* Coefficients are just 1/3, 1/4, 1/5 and 1/6 */
  float poly = __clc_mad(
      r,
      __clc_mad(r,
                __clc_mad(r, __clc_mad(r, 0x1.24924ap-3f, 0x1.555556p-3f),
                          0x1.99999ap-3f),
                0x1.000000p-2f),
      0x1.555556p-2f);

  poly *= r2 * r;

  float lth_near1 = -r2 * 0.5f;
  float ltt_near1 = -poly;
  float lt_near1 = lth_near1 + ltt_near1;
  float lh_near1 = -r;
  float l_near1 = lh_near1 + lt_near1;

  /* Computations for x not near 1 */
  int m = (int)(ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
  float mf = (float)m;
  int ixs = __clc_as_int(__clc_as_float(ax | 0x3f800000) - 1.0f);
  float mfs = (float)((ixs >> EXPSHIFTBITS_SP32) - 253);
  int c = m == -127;
  int ixn = c ? ixs : ax;
  float mfn = c ? mfs : mf;

  int indx = (ixn & 0x007f0000) + ((ixn & 0x00008000) << 1);

  /* F - Y */
  float f = __clc_as_float(0x3f000000 | indx) -
            __clc_as_float(0x3f000000 | (ixn & MANTBITS_SP32));

  indx = indx >> 16;
  float2 tv;
  tv.s0 = USE_TABLE(log_inv_tbl_ep_head, indx);
  tv.s1 = USE_TABLE(log_inv_tbl_ep_tail, indx);
  float rh = f * tv.s0;
  float rt = f * tv.s1;
  r = rh + rt;

  poly = __clc_mad(r, __clc_mad(r, 0x1.0p-2f, 0x1.555556p-2f), 0x1.0p-1f) *
         (r * r);
  poly += (rh - r) + rt;

  const float LOG2_HEAD = 0x1.62e000p-1f;  /* 0.693115234 */
  const float LOG2_TAIL = 0x1.0bfbe8p-15f; /* 0.0000319461833 */
  tv.s0 = USE_TABLE(loge_tbl_lo, indx);
  tv.s1 = USE_TABLE(loge_tbl_hi, indx);
  float lth = -r;
  float ltt = __clc_mad(mfn, LOG2_TAIL, -poly) + tv.s1;
  float lt = lth + ltt;
  float lh = __clc_mad(mfn, LOG2_HEAD, tv.s0);
  float l = lh + lt;

  /* Select near 1 or not */
  lth = near1 ? lth_near1 : lth;
  ltt = near1 ? ltt_near1 : ltt;
  lt = near1 ? lt_near1 : lt;
  lh = near1 ? lh_near1 : lh;
  l = near1 ? l_near1 : l;

  float gh = __clc_as_float(__clc_as_int(l) & 0xfffff000);
  float gt = ((ltt - (lt - lth)) + ((lh - l) + lt)) + (l - gh);

  float yh = __clc_as_float(iy & 0xfffff000);

  float yt = y - yh;

  float ylogx_s = __clc_mad(gt, yh, __clc_mad(gh, yt, yt * gt));
  float ylogx = __clc_mad(yh, gh, ylogx_s);
  float ylogx_t = __clc_mad(yh, gh, -ylogx) + ylogx_s;

  /* Extra precise exp of ylogx */
  /* 64/log2 : 92.332482616893657 */
  const float R_64_BY_LOG2 = 0x1.715476p+6f;
  int n = __clc_convert_int(ylogx * R_64_BY_LOG2);
  float nf = (float)n;

  int j = n & 0x3f;
  m = n >> 6;
  int m2 = m << EXPSHIFTBITS_SP32;

  /* log2/64 lead: 0.0108032227 */
  const float R_LOG2_BY_64_LD = 0x1.620000p-7f;
  /* log2/64 tail: 0.0000272020388 */
  const float R_LOG2_BY_64_TL = 0x1.c85fdep-16f;
  r = __clc_mad(nf, -R_LOG2_BY_64_TL, __clc_mad(nf, -R_LOG2_BY_64_LD, ylogx)) +
      ylogx_t;

  /* Truncated Taylor series for e^r */
  poly = __clc_mad(__clc_mad(__clc_mad(r, 0x1.555556p-5f, 0x1.555556p-3f), r,
                             0x1.000000p-1f),
                   r * r, r);

  tv.s0 = USE_TABLE(exp_tbl_ep_head, j);
  tv.s1 = USE_TABLE(exp_tbl_ep_tail, j);

  float expylogx =
      __clc_mad(tv.s0, poly, __clc_mad(tv.s1, poly, tv.s1)) + tv.s0;
  float sexpylogx = expylogx * __clc_as_float(0x1 << (m + 149));
  float texpylogx = __clc_as_float(__clc_as_int(expylogx) + m2);
  expylogx = m < -125 ? sexpylogx : texpylogx;

  /* Result is +-Inf if (ylogx + ylogx_t) > 128*log2 */
  expylogx = (ylogx > 0x1.62e430p+6f) |
                     (ylogx == 0x1.62e430p+6f & ylogx_t > -0x1.05c610p-22f)
                 ? __clc_as_float(PINFBITPATT_SP32)
                 : expylogx;

  /* Result is 0 if ylogx < -149*log2 */
  expylogx = ylogx < -0x1.9d1da0p+6f ? 0.0f : expylogx;

  /* Classify y:
   *   inty = 0 means not an integer.
   *   inty = 1 means odd integer.
   *   inty = 2 means even integer.
   */

  int yexp = (int)(ay >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32 + 1;
  int mask = (1 << (24 - yexp)) - 1;
  int yodd = ((iy >> (24 - yexp)) & 0x1) != 0;
  int inty = yodd ? 1 : 2;
  inty = (iy & mask) != 0 ? 0 : inty;
  inty = yexp < 1 ? 0 : inty;
  inty = yexp > 24 ? 2 : inty;

  float signval = __clc_as_float((__clc_as_uint(expylogx) ^ SIGNBIT_SP32));
  expylogx = ((inty == 1) & !xpos) ? signval : expylogx;
  int ret = __clc_as_int(expylogx);

  /* Corner case handling */
  ret = (!xpos & (inty == 0)) ? QNANBITPATT_SP32 : ret;
  ret = ax < 0x3f800000 & iy == NINFBITPATT_SP32 ? PINFBITPATT_SP32 : ret;
  ret = ax > 0x3f800000 & iy == NINFBITPATT_SP32 ? 0 : ret;
  ret = ax < 0x3f800000 & iy == PINFBITPATT_SP32 ? 0 : ret;
  ret = ax > 0x3f800000 & iy == PINFBITPATT_SP32 ? PINFBITPATT_SP32 : ret;
  int xinf = xpos ? PINFBITPATT_SP32 : NINFBITPATT_SP32;
  ret = ((ax == 0) & !ypos & (inty == 1)) ? xinf : ret;
  ret = ((ax == 0) & !ypos & (inty != 1)) ? PINFBITPATT_SP32 : ret;
  int xzero = xpos ? 0 : 0x80000000;
  ret = ((ax == 0) & ypos & (inty == 1)) ? xzero : ret;
  ret = ((ax == 0) & ypos & (inty != 1)) ? 0 : ret;
  ret = ((ax == 0) & (iy == NINFBITPATT_SP32)) ? PINFBITPATT_SP32 : ret;
  ret = ((ix == 0xbf800000) & (ay == PINFBITPATT_SP32)) ? 0x3f800000 : ret;
  ret = ((ix == NINFBITPATT_SP32) & !ypos & (inty == 1)) ? 0x80000000 : ret;
  ret = ((ix == NINFBITPATT_SP32) & !ypos & (inty != 1)) ? 0 : ret;
  ret =
      ((ix == NINFBITPATT_SP32) & ypos & (inty == 1)) ? NINFBITPATT_SP32 : ret;
  ret =
      ((ix == NINFBITPATT_SP32) & ypos & (inty != 1)) ? PINFBITPATT_SP32 : ret;
  ret = ((ix == PINFBITPATT_SP32) & !ypos) ? 0 : ret;
  ret = ((ix == PINFBITPATT_SP32) & ypos) ? PINFBITPATT_SP32 : ret;
  ret = (ax > PINFBITPATT_SP32) ? ix : ret;
  ret = (ay > PINFBITPATT_SP32) ? iy : ret;
  ret = ay == 0 ? 0x3f800000 : ret;
  ret = ix == 0x3f800000 ? 0x3f800000 : ret;

  return __clc_as_float(ret);
}
_CLC_BINARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_pow, float, float)

#ifdef cl_khr_fp64
_CLC_DEF _CLC_OVERLOAD double __clc_pow(double x, double y) {
  const double real_log2_tail = 5.76999904754328540596e-08;
  const double real_log2_lead = 6.93147122859954833984e-01;

  long ux = __clc_as_long(x);
  long ax = ux & (~SIGNBIT_DP64);
  int xpos = ax == ux;

  long uy = __clc_as_long(y);
  long ay = uy & (~SIGNBIT_DP64);
  int ypos = ay == uy;

  // Extended precision log
  double v, vt;
  {
    int exp = (int)(ax >> 52) - 1023;
    int mask_exp_1023 = exp == -1023;
    double xexp = (double)exp;
    long mantissa = ax & 0x000FFFFFFFFFFFFFL;

    long temp_ux =
        __clc_as_long(__clc_as_double(0x3ff0000000000000L | mantissa) - 1.0);
    exp = ((temp_ux & 0x7FF0000000000000L) >> 52) - 2045;
    double xexp1 = (double)exp;
    long mantissa1 = temp_ux & 0x000FFFFFFFFFFFFFL;

    xexp = mask_exp_1023 ? xexp1 : xexp;
    mantissa = mask_exp_1023 ? mantissa1 : mantissa;

    long rax = (mantissa & 0x000ff00000000000) +
               ((mantissa & 0x0000080000000000) << 1);
    int index = rax >> 44;

    double F = __clc_as_double(rax | 0x3FE0000000000000L);
    double Y = __clc_as_double(mantissa | 0x3FE0000000000000L);
    double f = F - Y;
    double2 tv;
    tv.s0 = USE_TABLE(log_f_inv_tbl_head, index);
    tv.s1 = USE_TABLE(log_f_inv_tbl_tail, index);
    double log_h = tv.s0;
    double log_t = tv.s1;
    double f_inv = (log_h + log_t) * f;
    double r1 = __clc_as_double(__clc_as_long(f_inv) & 0xfffffffff8000000L);
    double r2 = __spirv_ocl_fma(-F, r1, f) * (log_h + log_t);
    double r = r1 + r2;

    double poly = __clc_fma(
        r,
        __clc_fma(r,
                  __clc_fma(r, __clc_fma(r, 1.0 / 7.0, 1.0 / 6.0), 1.0 / 5.0),
                  1.0 / 4.0),
        1.0 / 3.0);
    poly = poly * r * r * r;

    double hr1r1 = 0.5 * r1 * r1;
    double poly0h = r1 + hr1r1;
    double poly0t = r1 - poly0h + hr1r1;
    poly = __clc_fma(r1, r2, __clc_fma(0.5 * r2, r2, poly)) + r2 + poly0t;

    tv.s0 = USE_TABLE(powlog_tbl_head, index);
    tv.s1 = USE_TABLE(powlog_tbl_tail, index);
    log_h = tv.s0;
    log_t = tv.s1;

    double resT_t = __clc_fma(xexp, real_log2_tail, +log_t) - poly;
    double resT = resT_t - poly0h;
    double resH = __clc_fma(xexp, real_log2_lead, log_h);
    double resT_h = poly0h;

    double H = resT + resH;
    double H_h = __clc_as_double(__clc_as_long(H) & 0xfffffffff8000000L);
    double T = (resH - H + resT) + (resT_t - (resT + resT_h)) + (H - H_h);
    H = H_h;

    double y_head = __clc_as_double(uy & 0xfffffffff8000000L);
    double y_tail = y - y_head;

    double temp = __clc_fma(y_tail, H, __clc_fma(y_head, T, y_tail * T));
    v = __clc_fma(y_head, H, temp);
    vt = __clc_fma(y_head, H, -v) + temp;
  }

  // Now calculate exp of (v,vt)

  double expv;
  {
    const double max_exp_arg = 709.782712893384;
    const double min_exp_arg = -745.1332191019411;
    const double sixtyfour_by_lnof2 = 92.33248261689366;
    const double lnof2_by_64_head = 0.010830424260348081;
    const double lnof2_by_64_tail = -4.359010638708991e-10;

    double temp = v * sixtyfour_by_lnof2;
    int n = (int)temp;
    double dn = (double)n;
    int j = n & 0x0000003f;
    int m = n >> 6;

    double2 tv;
    tv.s0 = USE_TABLE(two_to_jby64_ep_tbl_head, j);
    tv.s1 = USE_TABLE(two_to_jby64_ep_tbl_tail, j);
    double f1 = tv.s0;
    double f2 = tv.s1;
    double f = f1 + f2;

    double r1 = __clc_fma(dn, -lnof2_by_64_head, v);
    double r2 = dn * lnof2_by_64_tail;
    double r = (r1 + r2) + vt;

    double q =
        __clc_fma(r,
                  __clc_fma(r,
                            __clc_fma(r,
                                      __clc_fma(r, 1.38889490863777199667e-03,
                                                8.33336798434219616221e-03),
                                      4.16666666662260795726e-02),
                            1.66666666665260878863e-01),
                  5.00000000000000008883e-01);
    q = __clc_fma(r * r, q, r);

    expv = __clc_fma(f, q, f2) + f1;
    expv = __spirv_ocl_ldexp(expv, m);

    expv = v > max_exp_arg ? __clc_as_double(0x7FF0000000000000L) : expv;
    expv = v < min_exp_arg ? 0.0 : expv;
  }

  // See whether y is an integer.
  // inty = 0 means not an integer.
  // inty = 1 means odd integer.
  // inty = 2 means even integer.

  int inty;
  {
    int yexp = (int)(ay >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64 + 1;
    inty = yexp < 1 ? 0 : 2;
    inty = yexp > 53 ? 2 : inty;
    long mask = (1L << (53 - yexp)) - 1L;
    int inty1 = (((ay & ~mask) >> (53 - yexp)) & 1L) == 1L ? 1 : 2;
    inty1 = (ay & mask) != 0 ? 0 : inty1;
    inty = !(yexp < 1) & !(yexp > 53) ? inty1 : inty;
  }

  expv *= (inty == 1) & !xpos ? -1.0 : 1.0;

  long ret = __clc_as_long(expv);

  // Now all the edge cases
  ret = !xpos & (inty == 0) ? QNANBITPATT_DP64 : ret;
  ret = ax < 0x3ff0000000000000L & uy == NINFBITPATT_DP64 ? PINFBITPATT_DP64
                                                          : ret;
  ret = ax > 0x3ff0000000000000L & uy == NINFBITPATT_DP64 ? 0L : ret;
  ret = ax < 0x3ff0000000000000L & uy == PINFBITPATT_DP64 ? 0L : ret;
  ret = ax > 0x3ff0000000000000L & uy == PINFBITPATT_DP64 ? PINFBITPATT_DP64
                                                          : ret;
  long xinf = xpos ? PINFBITPATT_DP64 : NINFBITPATT_DP64;
  ret = ((ax == 0L) & !ypos & (inty == 1)) ? xinf : ret;
  ret = ((ax == 0L) & !ypos & (inty != 1)) ? PINFBITPATT_DP64 : ret;
  long xzero = xpos ? 0L : 0x8000000000000000L;
  ret = ((ax == 0L) & ypos & (inty == 1)) ? xzero : ret;
  ret = ((ax == 0L) & ypos & (inty != 1)) ? 0L : ret;
  ret = ((ax == 0L) & (uy == NINFBITPATT_DP64)) ? PINFBITPATT_DP64 : ret;
  ret = ((ux == 0xbff0000000000000L) & (ay == PINFBITPATT_DP64))
            ? 0x3ff0000000000000L
            : ret;
  ret = ((ux == NINFBITPATT_DP64) & !ypos & (inty == 1)) ? 0x8000000000000000L
                                                         : ret;
  ret = ((ux == NINFBITPATT_DP64) & !ypos & (inty != 1)) ? 0L : ret;
  ret =
      ((ux == NINFBITPATT_DP64) & ypos & (inty == 1)) ? NINFBITPATT_DP64 : ret;
  ret =
      ((ux == NINFBITPATT_DP64) & ypos & (inty != 1)) ? PINFBITPATT_DP64 : ret;
  ret = (ux == PINFBITPATT_DP64) & !ypos ? 0L : ret;
  ret = (ux == PINFBITPATT_DP64) & ypos ? PINFBITPATT_DP64 : ret;
  ret = ax > PINFBITPATT_DP64 ? ux : ret;
  ret = ay > PINFBITPATT_DP64 ? uy : ret;
  ret = ay == 0L ? 0x3ff0000000000000L : ret;
  ret = ux == 0x3ff0000000000000L ? 0x3ff0000000000000L : ret;

  return __clc_as_double(ret);
}
_CLC_BINARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, double, __clc_pow, double, double)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half __clc_pow(half x, half y) {
  return __clc_pow((float)x, (float)y);
}

_CLC_BINARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, half, __clc_pow, half, half)
#endif
