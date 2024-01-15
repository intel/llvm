
//==------ imf_rounding_op.hpp - simple fp op with rounding mode support----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#ifndef __LIBDEVICE_IMF_ROUNDING_OP_H__
#define __LIBDEVICE_IMF_ROUNDING_OP_H__
#include "imf_impl_utils.hpp"
#include <limits>
template <typename Ty>
static Ty __handling_fp_overflow(unsigned z_sig, int rd) {
  typedef typename __iml_fp_config<Ty>::utype UTy;
  UTy temp;
  if (z_sig == 0) {
    if (rd == __IML_RTZ || rd == __IML_RTN) {
      temp = __iml_fp_config<Ty>::max_fin_bits;
    } else {
      temp = __iml_fp_config<Ty>::pos_inf_bits;
    }
  } else {
    if (rd == __IML_RTZ || rd == __IML_RTP) {
      temp = __iml_fp_config<Ty>::min_fin_bits;
    } else {
      temp = __iml_fp_config<Ty>::neg_inf_bits;
    }
  }
  return __builtin_bit_cast(Ty, temp);
}

template <typename Ty, typename UTy>
static Ty __handling_fp_underflow(UTy z_sig, int rd, bool above_half) {
  if (z_sig == 0) {
    if ((rd == __IML_RTP) || ((rd == __IML_RTE) && above_half))
      return __builtin_bit_cast(Ty, static_cast<UTy>(0x1));
    else
      return __builtin_bit_cast(Ty, static_cast<UTy>(0x0));
  } else {
    if ((rd == __IML_RTN) || ((rd == __IML_RTE) && above_half))
      return __builtin_bit_cast(Ty, z_sig << (sizeof(Ty) * 8 - 1) | 0x1);
    else
      return __builtin_bit_cast(Ty, z_sig << (sizeof(Ty) * 8 - 1));
  }
}

template <typename UTy>
static UTy __handling_rounding(UTy sig, UTy fra, unsigned grs, int rd) {
  if (grs == 0)
    return 0;
  if ((__IML_RTP == rd) && (sig == 0))
    return 1;
  if ((__IML_RTN == rd) && (sig == 1))
    return 1;
  if ((__IML_RTE == rd)) {
    if ((grs > 0x4) || ((grs == 0x4) && ((fra & 0x1) == 0x1)))
      return 1;
  }
  return 0;
}

// Pre-assumption, x's exp >= y's exp.
template <typename Ty> Ty __fp_add_sig_same(Ty x, Ty y, int rd) {
  typedef typename __iml_fp_config<Ty>::utype UTy;
  UTy x_bit = __builtin_bit_cast(UTy, x);
  UTy y_bit = __builtin_bit_cast(UTy, y);
  UTy x_exp = (x_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy y_exp = (y_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy x_fra = x_bit & __iml_fp_config<Ty>::fra_mask;
  UTy y_fra = y_bit & __iml_fp_config<Ty>::fra_mask;
  UTy x_sig = x_bit >> ((sizeof(Ty) * 8) - 1);
  UTy z_exp, z_fra;
  unsigned y_fra_shift = 0;
  // x_ib, y_ib are implicit bit, 1 for normal value, 0 for subnormal value
  unsigned g_bit = 0, r_bit = 0, s_bit = 0, x_ib = 0, y_ib = 0;

  if (y_exp == 0) {
    if (x_exp == 0) {
      // case 1: adding 2 subnormal values, directly add x_fra and y_fra and
      // rise into normal value range when overflow happens.
      z_fra = x_fra + y_fra;
      if (z_fra > __iml_fp_config<Ty>::fra_mask) {
        z_fra = z_fra & __iml_fp_config<Ty>::fra_mask;
        z_exp = 1;
      } else
        z_exp = 0;
      return __builtin_bit_cast(
          Ty, (x_sig << ((sizeof(Ty) * 8) - 1)) |
                  (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
    }
    // case 2, x is normal value and y is subnormal value, should align y's exp
    // with x's. x is represented as 2^(x_exp - 127)*(1.x_fra) and y is
    // represented as 2^(-126)*(0.y_fra), the y_fra_shift should be:
    // (x_exp - 127) - (-126) = x_exp - 1;
    y_fra_shift = x_exp - 1;
    y_ib = 0;
    x_ib = 1;
  } else {
    // case 3, x and y are both normal value.
    y_fra_shift = x_exp - y_exp;
    x_ib = 1;
    y_ib = 1;
  }

  // y_fra_shift means y needs to be right shifted in order to align exp
  // with x, guard, round, sticky bit will be initialized and y_fra will
  // change to.
  if (y_fra_shift > 0) {
    if (y_fra_shift == 1) {
      // if y_fra_shift == 1, bit 0 of y_fra will be dicarded from frac
      // and be guard bit.
      g_bit = y_fra & 0x1;
      y_fra = y_fra >> 1;
      if (y_ib == 1)
        y_fra = y_fra |
                (static_cast<UTy>(1) << (std::numeric_limits<Ty>::digits - 2));
    } else if (y_fra_shift <= (std::numeric_limits<Ty>::digits - 1)) {
      // For fp32, when y_fra_shift <= 23, part of fra bits will be discarded
      // and these bits will be used to initialize g,r,s bit. Situation is
      // similar for fp64.
      g_bit = (y_fra & (static_cast<UTy>(0x1) << (y_fra_shift - 1))) ? 1 : 0;
      r_bit = (y_fra & (static_cast<UTy>(0x1) << (y_fra_shift - 2))) ? 1 : 0;
      s_bit =
          (y_fra & ((static_cast<UTy>(0x1) << (y_fra_shift - 2)) - 1)) ? 1 : 0;
      y_fra = y_fra >> y_fra_shift;
      if (y_ib == 1)
        y_fra =
            y_fra | (static_cast<UTy>(0x1)
                     << (std::numeric_limits<Ty>::digits - 1 - y_fra_shift));
    } else if (y_fra_shift == std::numeric_limits<Ty>::digits) {
      // For fp32, when y_fra_shift == 24, fra will be 0 and implicit bit will
      // be guard bit, bit 22 of original fra will be round bit. Situation is
      // similar for fp64.
      g_bit = y_ib;
      r_bit = (y_fra &
               (static_cast<UTy>(0x1) << (std::numeric_limits<Ty>::digits - 2)))
                  ? 1
                  : 0;
      s_bit = (y_fra & ((static_cast<UTy>(0x1)
                         << (std::numeric_limits<Ty>::digits - 2)) -
                        1))
                  ? 1
                  : 0;
      y_fra = 0;
    } else if (y_fra_shift == 25) {
      r_bit = y_ib;
      s_bit = (y_fra != 0) ? 1 : 0;
      y_fra = 0;
    } else {
      s_bit = ((y_bit != 0) || (y_fra != 0)) ? 1 : 0;
      y_fra = 0;
    }
    y_ib = 0;
  }

  z_exp = x_exp;
  z_fra = x_fra + y_fra;
  unsigned z_ib = 0;
  if (z_fra > __iml_fp_config<Ty>::fra_mask) {
    z_fra = z_fra & __iml_fp_config<Ty>::fra_mask;
    z_ib = x_ib + y_ib + 1;
    // z_ib >= 2, z_fra needs to right shift for 1 bit.
    s_bit = (r_bit == 1) ? 1 : s_bit;
    r_bit = g_bit;
    g_bit = z_fra & 0x1;
    z_fra = z_fra >> 1;
    if (z_ib == 3)
      z_fra = z_fra |
              (static_cast<UTy>(0x1) << (std::numeric_limits<Ty>::digits - 2));
    z_exp++;
  } else {
    z_ib = x_ib + y_ib;
    if (z_ib == 2) {
      s_bit = (r_bit == 1) ? 1 : s_bit;
      r_bit = g_bit;
      g_bit = z_fra & 0x1;
      z_fra = z_fra >> 1;
      z_exp++;
    }
  }

  UTy rb = __handling_rounding(x_sig, z_fra,
                               ((g_bit << 2) | (r_bit << 1) | s_bit), rd);
  z_fra += rb;
  if (z_fra ==
      (static_cast<UTy>(0x1) << (std::numeric_limits<Ty>::digits - 1))) {
    z_fra = 0;
    z_exp++;
  }

  if (z_exp == __iml_fp_config<Ty>::exp_mask)
    return __handling_fp_overflow<Ty>(x_sig, rd);
  return __builtin_bit_cast(
      Ty, x_sig << (sizeof(Ty) * 8 - 1) |
              (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
}

template <typename Ty> Ty __fp_add_sig_diff(Ty x, Ty y, int rd) {
  typedef typename __iml_fp_config<Ty>::utype UTy;
  UTy x_bit = __builtin_bit_cast(UTy, x);
  UTy y_bit = __builtin_bit_cast(UTy, y);
  UTy x_exp = (x_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy y_exp = (y_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy x_fra = x_bit & __iml_fp_config<Ty>::fra_mask;
  UTy y_fra = y_bit & __iml_fp_config<Ty>::fra_mask;
  UTy x_sig = x_bit >> ((sizeof(Ty) * 8) - 1);
  UTy y_sig = y_bit >> ((sizeof(Ty) * 8) - 1);
  UTy z_exp, z_fra, z_sig;
  unsigned y_fra_shift = 0;
  unsigned g_bit = 0, r_bit = 0, s_bit = 0, x_ib = 0, y_ib = 0;

  if (y_exp == 0) {
    if (x_exp == 0) {
      z_exp = 0;
      if (x_fra > y_fra) {
        z_sig = x_sig;
        z_fra = x_fra - y_fra;
      } else {
        z_fra = y_fra - x_fra;
        z_sig = y_sig;
      }
      return __builtin_bit_cast(
          Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                  (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
    }
    x_ib = 1;
    y_ib = 0;
    y_fra_shift = x_exp - 1;
  } else {
    x_ib = 1;
    y_ib = 1;
    y_fra_shift = x_exp - y_exp;
  }

  if (y_fra_shift > 0) {
    if (y_fra_shift == 1) {
      g_bit = y_fra & 0x1;
      y_fra = y_fra >> 1;
      if (y_ib == 1)
        y_fra = y_fra | (static_cast<UTy>(0x1)
                         << (std::numeric_limits<Ty>::digits - 2));
    } else if (y_fra_shift <= (std::numeric_limits<Ty>::digits - 1)) {
      g_bit = (y_fra & (static_cast<UTy>(0x1) << (y_fra_shift - 1))) ? 1 : 0;
      r_bit = (y_fra & (static_cast<UTy>(0x1) << (y_fra_shift - 2))) ? 1 : 0;
      s_bit =
          (y_fra & ((static_cast<UTy>(0x1) << (y_fra_shift - 2)) - 1)) ? 1 : 0;
      y_fra = y_fra >> y_fra_shift;
      if (y_ib == 1)
        y_fra =
            y_fra | (static_cast<UTy>(0x1)
                     << (std::numeric_limits<Ty>::digits - 1 - y_fra_shift));
    } else if (y_fra_shift == std::numeric_limits<Ty>::digits) {
      g_bit = y_ib;
      r_bit = (y_fra &
               (static_cast<UTy>(0x1) << (std::numeric_limits<Ty>::digits - 2)))
                  ? 1
                  : 0;
      s_bit = (y_fra & ((static_cast<UTy>(0x1)
                         << (std::numeric_limits<Ty>::digits - 2)) -
                        1))
                  ? 1
                  : 0;
      y_fra = 0;
    } else if (y_fra_shift == (std::numeric_limits<Ty>::digits + 1)) {
      r_bit = y_ib;
      s_bit = (y_fra != 0) ? 1 : 0;
      y_fra = 0;
    } else {
      s_bit = ((y_bit != 0) || (y_fra != 0)) ? 1 : 0;
      y_fra = 0;
    }
    y_ib = 0;
  } else {
    z_exp = x_exp;
    if (x_fra == y_fra) {
      return (y_exp != 0)
                 ? __builtin_bit_cast(Ty, static_cast<UTy>(0x0))
                 : __builtin_bit_cast(
                       Ty, (x_sig << (sizeof(Ty) * 8 - 1)) |
                               (static_cast<UTy>(0x1)
                                << (std::numeric_limits<Ty>::digits - 1)));
    } else if (x_fra > y_fra) {
      z_fra = x_fra - y_fra;
      z_sig = x_sig;
      if (y_exp == 0)
        return __builtin_bit_cast(
            Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                    (static_cast<UTy>(0x1)
                     << (std::numeric_limits<Ty>::digits - 1)) |
                    z_fra);
    } else {
      if (y_exp != 0) {
        z_fra = y_fra - x_fra;
        z_sig = y_sig;
      } else {
        // Don't need to do normalization here
        z_fra = (x_fra | (static_cast<UTy>(0x1)
                          << (std::numeric_limits<Ty>::digits - 1))) -
                y_fra;
        return __builtin_bit_cast(Ty, (x_sig << (sizeof(Ty) * 8 - 1)) | z_fra);
      }
    }
    int norm_shift =
        get_leading_zeros_from(z_fra, std::numeric_limits<Ty>::digits - 1);
    if (z_exp > (norm_shift + 1)) {
      z_exp = z_exp - norm_shift - 1;
      z_fra = (z_fra << (norm_shift + 1)) & __iml_fp_config<Ty>::fra_mask;
    } else {
      z_fra = z_fra << (z_exp - 1);
      z_exp = 0;
    }
    return __builtin_bit_cast(
        Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
  }

  z_exp = x_exp;
  z_sig = x_sig;
  x_fra = x_fra << 3;
  y_fra = y_fra << 3;
  y_fra = y_fra | (g_bit << 2) | (r_bit << 1) | s_bit;
  unsigned z_grs;
  if (x_fra >= y_fra) {
    z_fra = x_fra - y_fra;
    z_grs = z_fra & 0x7;
    z_fra = z_fra >> 3;
  } else {
    x_fra = x_fra |
            (static_cast<UTy>(0x1) << (std::numeric_limits<Ty>::digits + 2));
    // x_fra, y_fra, z_fra are 26 bit for fp32 and 55 bit for fp64.
    z_fra = x_fra - y_fra;
    int norm_shift =
        get_leading_zeros_from(z_fra, std::numeric_limits<Ty>::digits + 2);
    if ((y_fra_shift == 1) && (y_exp != 0)) {
      // In this case, original rounding and sticky bit is 0, z_fra tailing 3
      // bit is 000 or 100 and grounding bit must be shifted to final z
      // mantissa, so don't need to care about final rounding.
      if (z_exp > (norm_shift + 1)) {
        z_exp = z_exp - norm_shift - 1;
        // The most significant bit 1 should be discarded since it will be
        // implicit bit and tailing 3 bits should be discarded too.
        z_fra = ((z_fra << (norm_shift + 1)) &
                 (__iml_fp_config<Ty>::fra_mask << 3)) >>
                3;
      } else {
        // falls into subnormal values, no implicit bit needed.
        z_fra =
            ((z_fra << (z_exp - 1)) & (__iml_fp_config<Ty>::fra_mask << 3)) >>
            3;
        z_exp = 0;
      }
      return __builtin_bit_cast(
          Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                  (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
    } else {
      // In this case, z_fra bit 25 is 1, so norm shift 1 bit, original guard
      // bit will be bit of mantissa.
      z_grs = z_fra & 0x7;
      z_fra = (z_fra & (__iml_fp_config<Ty>::fra_mask << 2)) >> 2;
      z_grs = (z_grs & 0x3) << 1;
      z_exp -= 1;
    }
  }
  int rb = __handling_rounding(z_sig, z_fra, z_grs, rd);
  z_fra = z_fra + rb;
  if (z_fra > __iml_fp_config<Ty>::fra_mask) {
    z_fra = 0;
    z_exp += 1;
  }
  return __builtin_bit_cast(
      Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
              (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
}

template <typename Ty> Ty __fp_add_sub_entry(Ty x, Ty y, int rd) {
  typedef typename __iml_fp_config<Ty>::utype UTy;
  UTy x_bit = __builtin_bit_cast(UTy, x);
  UTy y_bit = __builtin_bit_cast(UTy, y);
  UTy x_exp = (x_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy y_exp = (y_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy x_fra = x_bit & __iml_fp_config<Ty>::fra_mask;
  UTy y_fra = y_bit & __iml_fp_config<Ty>::fra_mask;
  UTy x_sig = x_bit >> ((sizeof(Ty) * 8) - 1);
  UTy y_sig = y_bit >> ((sizeof(Ty) * 8) - 1);
  UTy temp;
  if (((x_exp == __iml_fp_config<Ty>::exp_mask) && (x_fra != 0)) ||
      ((y_exp == __iml_fp_config<Ty>::exp_mask) && (y_fra != 0))) {
    temp = __iml_fp_config<Ty>::nan_bits;
    return __builtin_bit_cast(Ty, temp);
  }

  if ((x_exp == __iml_fp_config<Ty>::exp_mask) && (x_fra == 0)) {
    if ((y_exp == __iml_fp_config<Ty>::exp_mask) && (y_fra == 0)) {
      if (x_sig != y_sig) {
        temp = __iml_fp_config<Ty>::nan_bits;
        return __builtin_bit_cast(Ty, temp);
      }
    }
    return x;
  }

  if ((y_exp == __iml_fp_config<Ty>::exp_mask) && (y_fra == 0))
    return y;
  if ((x_exp == 0) && (x_fra == 0))
    return y;
  if ((y_exp == 0) && (y_fra == 0))
    return x;

  if (x_sig == y_sig)
    return (x_exp > y_exp) ? __fp_add_sig_same(x, y, rd)
                           : __fp_add_sig_same(y, x, rd);
  else
    return (x_exp > y_exp) ? __fp_add_sig_diff(x, y, rd)
                           : __fp_add_sig_diff(y, x, rd);
}

template <typename Ty> Ty __fp_mul(Ty x, Ty y, int rd) {
  typedef typename __iml_fp_config<Ty>::utype UTy;
  typedef typename __iml_get_double_size_unsigned<UTy>::utype DSUTy;
  UTy x_bit = __builtin_bit_cast(UTy, x);
  UTy y_bit = __builtin_bit_cast(UTy, y);
  UTy x_exp = (x_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy y_exp = (y_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy x_fra = x_bit & __iml_fp_config<Ty>::fra_mask;
  UTy y_fra = y_bit & __iml_fp_config<Ty>::fra_mask;
  UTy x_sig = x_bit >> ((sizeof(Ty) * 8) - 1);
  UTy y_sig = y_bit >> ((sizeof(Ty) * 8) - 1);
  UTy z_sig = x_sig ^ y_sig;
  UTy z_exp, z_fra;
  UTy x_ib, y_ib;
  int z_exp_s = 0;

  if (((x_exp == __iml_fp_config<Ty>::exp_mask) && (x_fra != 0)) ||
      ((y_exp == __iml_fp_config<Ty>::exp_mask) && (y_fra != 0))) {
    UTy temp = __iml_fp_config<Ty>::nan_bits;
    return __builtin_bit_cast(Ty, temp);
  }

  if (((x_exp == __iml_fp_config<Ty>::exp_mask) && (x_fra == 0)) ||
      ((y_exp == __iml_fp_config<Ty>::exp_mask) && (y_fra == 0))) {
    return __builtin_bit_cast(Ty,
                              (z_sig << (sizeof(Ty) * 8 - 1)) |
                                  (__iml_fp_config<Ty>::exp_mask
                                   << (std::numeric_limits<Ty>::digits - 1)));
  }

  if ((x_exp == 0x0) && (x_fra == 0x0))
    return __builtin_bit_cast(Ty, static_cast<UTy>(0x0));
  if ((y_exp == 0x0) && (y_fra == 0x0))
    return __builtin_bit_cast(Ty, static_cast<UTy>(0x0));

  if (x_exp == 0x0) {
    x_ib = 0;
    z_exp_s -= (__iml_fp_config<Ty>::bias - 1);
  } else {
    x_ib = 1;
    z_exp_s += static_cast<int>(x_exp) - __iml_fp_config<Ty>::bias;
  }

  if (y_exp == 0x0) {
    y_ib = 0;
    z_exp_s -= (__iml_fp_config<Ty>::bias - 1);
  } else {
    y_ib = 1;
    z_exp_s += static_cast<int>(y_exp) - __iml_fp_config<Ty>::bias;
  }

  if (z_exp_s >= __iml_fp_config<Ty>::bias + 1)
    return __handling_fp_overflow<Ty>(static_cast<unsigned>(z_sig), rd);

  // subnormal value multiplication will lead to underflow
  if ((x_ib == 0) && (y_ib == 0))
    return __handling_fp_underflow<Ty>(z_sig, rd, false);

  DSUTy x_ib_fra = static_cast<DSUTy>(
      (x_ib << (std::numeric_limits<Ty>::digits - 1)) | x_fra);
  DSUTy y_ib_fra = static_cast<DSUTy>(
      (y_ib << (std::numeric_limits<Ty>::digits - 1)) | y_fra);
  DSUTy z_ib_fra = x_ib_fra * y_ib_fra;
  unsigned g_bit = 0, r_bit = 0, s_bit = 0;

  // if z_ib_fra == 0, x_ib_fra or y_ib_fra is 0, x_ib or y_ib is
  // 0, then x or y is subnormal value and the x_fra or y_fra is
  // also 0, then the input is zero, final value is zero. Such case
  // has already been handled, so z_ib_fra is NON-zero and final
  // product can be represented: z_ib_fra * 2^-46 * 2^(z_exp_s) for
  // fp32 and the situation is same for fp64.
  size_t msb_pos;
  if constexpr (std::is_same<DSUTy, __iml_ui128>::value)
    msb_pos = z_ib_fra.ui128_msb_pos();
  else
    msb_pos = get_msb_pos(z_ib_fra);

  // Final product can be represented: z_ib_fra * 2^-46 * 2^(z_exp_s)
  // for fp32 and for fp64, final product can be represented as:
  // z_ib_fra * 2^-104 * 2^(z_exp_s)
  // 1. Try to handle final product as normal value:
  // 1.xxx... * 2^(msb_pos) * 2^(-46) * 2^(z_exp_s) for fp32
  // 1.xxx... * 2^(msb_pos) * 2^(-104) * 2^(z_exp_s) for fp64
  int tmp_exp = static_cast<int>(msb_pos) -
                (std::numeric_limits<Ty>::digits - 1) * 2 + z_exp_s;
  if (tmp_exp > __iml_fp_config<Ty>::bias)
    return __handling_fp_overflow<Ty>(static_cast<unsigned>(z_sig), rd);

  if (tmp_exp >= (1 - __iml_fp_config<Ty>::bias)) {
    z_exp = static_cast<UTy>(tmp_exp + __iml_fp_config<Ty>::bias);
    if (msb_pos <= (std::numeric_limits<Ty>::digits - 1)) {
      // no rounding mode needs to be considered here, all bits will be taken
      // into final fra
      z_fra = static_cast<UTy>(z_ib_fra &
                               ((static_cast<DSUTy>(0x1) << msb_pos) - 1));
      z_fra = z_fra << (std::numeric_limits<Ty>::digits - 1 - msb_pos);
      return __builtin_bit_cast(
          Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                  (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
    } else if (msb_pos == std::numeric_limits<Ty>::digits) {
      g_bit = static_cast<unsigned>(z_ib_fra & 0x1);
      z_fra = static_cast<UTy>(z_ib_fra);
      z_fra = (z_fra >> 1) & __iml_fp_config<Ty>::fra_mask;
    } else if (msb_pos == (std::numeric_limits<Ty>::digits + 1)) {
      r_bit = static_cast<unsigned>(z_ib_fra & 0x1);
      g_bit = static_cast<unsigned>((z_ib_fra & 0x2) >> 1);
      z_fra = static_cast<UTy>(z_ib_fra);
      z_fra = (z_fra >> 2) & __iml_fp_config<Ty>::fra_mask;
    } else {
      unsigned bit_discarded = msb_pos - (std::numeric_limits<Ty>::digits - 1);
      g_bit = static_cast<unsigned>(
          (z_ib_fra & (static_cast<DSUTy>(0x1) << (bit_discarded - 1))) >>
          (bit_discarded - 1));
      r_bit = static_cast<unsigned>(
          (z_ib_fra & (static_cast<DSUTy>(0x1) << (bit_discarded - 2))) >>
          (bit_discarded - 2));
      s_bit = ((z_ib_fra &
                ((static_cast<DSUTy>(0x1) << (bit_discarded - 2)) - 1)) != 0x0)
                  ? 1
                  : 0;
      UTy temp = __iml_fp_config<Ty>::fra_mask;
      z_fra = static_cast<UTy>((z_ib_fra >> bit_discarded) & temp);
    }

    int rb = __handling_rounding(z_sig, z_fra,
                                 ((g_bit << 2) | (r_bit << 1) | s_bit), rd);
    z_fra += rb;
    if (z_fra > __iml_fp_config<Ty>::fra_mask) {
      z_fra = 0;
      z_exp++;
    }
    if (z_exp == __iml_fp_config<Ty>::exp_mask)
      return __handling_fp_overflow<Ty>(static_cast<unsigned>(z_sig), rd);
    return __builtin_bit_cast(
        Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
  }

  // For fp32, If tmp_exp < -126, subnormal_exp_diff >= 1 and if
  // subnormal_exp_diff >= 24, underflow happens.
  // For fp64, if tmp_exp < -1022, subnormal_exp_diff >= 1 and if
  // subnormal_exp_diff >= 53, underflow happens.

  unsigned subnormal_exp_diff = (1 - __iml_fp_config<Ty>::bias) - tmp_exp;
  if (subnormal_exp_diff >= (std::numeric_limits<Ty>::digits)) {
    bool above_half = false;
    if (subnormal_exp_diff == std::numeric_limits<Ty>::digits) {
      // In this case, the most significant bit 1 will be guard bit, if rouding
      // and sticky bit are not zero, above_half is true.
      DSUTy z_t = z_ib_fra & ((static_cast<DSUTy>(0x1) << msb_pos) - 1);
      if (z_t > 0x0)
        above_half = true;
    }
    return __handling_fp_underflow<Ty>(z_sig, rd, above_half);
  }

  // For fp32, if subnormal_exp_diff <= 23, we will have subnormal_exp_diff - 1
  // leading 0 bit in final fra and take (24 - subnormal_exp_diff) bits starting
  // from msb_pos bit 1 in z_ib_fra.
  // For fp64, if subnormal_exp_diff <= 52, we will have subnormal_exp_diff - 1
  // leading 0 bit in final fra and take (53 - subnormal_exp_diff) bits starting
  // from msb_pos bit 1 in z_ib_fra.
  z_exp = 0x0;
  if ((std::numeric_limits<Ty>::digits - subnormal_exp_diff) >= (msb_pos + 1)) {
    // no bit discarded, no need for rounding handling.
    z_fra = static_cast<UTy>(z_ib_fra);
    z_fra = z_fra << ((std::numeric_limits<Ty>::digits - subnormal_exp_diff) -
                      (msb_pos + 1));
  } else {
    unsigned bit_discarded =
        (msb_pos + 1) - (std::numeric_limits<Ty>::digits - subnormal_exp_diff);
    if (bit_discarded == 1) {
      g_bit = static_cast<unsigned>(z_ib_fra & 0x1);
    } else if (bit_discarded == 2) {
      g_bit = static_cast<unsigned>((z_ib_fra & 0x2) >> 1);
      r_bit = static_cast<unsigned>(z_ib_fra & 0x1);
    } else {
      g_bit = static_cast<unsigned>(
          (z_ib_fra & (static_cast<DSUTy>(0x1) << (bit_discarded - 1))) >>
          (bit_discarded - 1));
      r_bit = static_cast<unsigned>(
          (z_ib_fra & (static_cast<DSUTy>(0x1) << (bit_discarded - 2))) >>
          (bit_discarded - 2));
      s_bit = ((z_ib_fra &
                ((static_cast<DSUTy>(0x1) << (bit_discarded - 2)) - 1)) != 0x0)
                  ? 1
                  : 0;
    }
    z_ib_fra = z_ib_fra >> bit_discarded;
    z_fra = static_cast<UTy>(z_ib_fra);
    int rb = __handling_rounding(z_sig, z_fra,
                                 (g_bit << 2) | (r_bit << 1) | s_bit, rd);
    z_fra += rb;
    if (z_fra > __iml_fp_config<Ty>::fra_mask) {
      z_fra = 0;
      z_exp++;
    }
  }
  return __builtin_bit_cast(
      Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
              (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
}

template <typename UTy> static UTy fra_uint_div(UTy x, UTy y, unsigned nbits) {
  UTy res = 0;
  unsigned iters = 0;
  if (x == 0)
    return 0x0;
  while (iters < nbits) {
    res = res << 1;
    x = x << 1;
    if (x > y) {
      x = x - y;
      res = res | 0x1;
    } else if (x == y) {
      res = res | 0x1;
      res = res << (nbits - iters - 1);
      return res;
    } else {
    }
    iters++;
  }
  res = res | 0x1;
  return res;
}

template <typename Ty> Ty __fp_div(Ty x, Ty y, int rd) {
  typedef typename __iml_fp_config<Ty>::utype UTy;
  typedef typename __iml_fp_config<Ty>::stype STy;
  UTy x_bit = __builtin_bit_cast(UTy, x);
  UTy y_bit = __builtin_bit_cast(UTy, y);
  UTy x_exp = (x_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy y_exp = (y_bit >> (std::numeric_limits<Ty>::digits - 1)) &
              __iml_fp_config<Ty>::exp_mask;
  UTy x_fra = x_bit & __iml_fp_config<Ty>::fra_mask;
  UTy y_fra = y_bit & __iml_fp_config<Ty>::fra_mask;
  UTy x_sig = x_bit >> ((sizeof(Ty) * 8) - 1);
  UTy y_sig = y_bit >> ((sizeof(Ty) * 8) - 1);
  UTy z_sig = x_sig ^ y_sig;
  UTy z_exp = 0x0, z_fra = 0x0;
  const UTy one_bits = 0x1;
  const UTy sig_off_mask = (one_bits << (sizeof(UTy) * 8 - 1)) - 1;

  if (((x_exp == __iml_fp_config<Ty>::exp_mask) && (x_fra != 0x0)) ||
      ((y_exp == __iml_fp_config<Ty>::exp_mask) && (y_fra != 0x0)) ||
      ((y_bit & sig_off_mask) == 0x0)) {
    UTy tmp = __iml_fp_config<Ty>::nan_bits;
    return __builtin_bit_cast(Ty, tmp);
  }

  if ((x_exp == __iml_fp_config<Ty>::exp_mask) && (x_fra == 0x0)) {
    if ((y_exp == __iml_fp_config<Ty>::exp_mask) && (y_fra == 0x0)) {
      UTy tmp = __iml_fp_config<Ty>::nan_bits;
      return __builtin_bit_cast(Ty, tmp);
    } else {
      UTy tmp =
          (z_sig << (sizeof(Ty) * 8 - 1)) | __iml_fp_config<Ty>::pos_inf_bits;
      return __builtin_bit_cast(Ty, tmp);
    }
  }

  if ((x_bit & sig_off_mask) == 0x0)
    return __builtin_bit_cast(Ty, (z_sig << (sizeof(UTy) * 8 - 1)) | 0x0);

  if ((y_exp == __iml_fp_config<Ty>::exp_mask) && (y_fra == 0x0))
    return __builtin_bit_cast(Ty, (z_sig << (sizeof(UTy) * 8 - 1)) | 0x0);

  int sx_exp = x_exp, sy_exp = y_exp;
  sx_exp = (sx_exp == 0) ? (1 - __iml_fp_config<Ty>::bias)
                         : (sx_exp - __iml_fp_config<Ty>::bias);
  sy_exp = (sy_exp == 0) ? (1 - __iml_fp_config<Ty>::bias)
                         : (sy_exp - __iml_fp_config<Ty>::bias);
  int exp_diff = sx_exp - sy_exp;
  if (x_exp != 0x0)
    x_fra = (one_bits << (std::numeric_limits<Ty>::digits - 1)) | x_fra;
  if (y_exp != 0x0)
    y_fra = (one_bits << (std::numeric_limits<Ty>::digits - 1)) | y_fra;

  if (x_fra >= y_fra) {
    // x_fra / y_fra max value for fp32 is 0xFFFFFF when x is normal
    // and y is subnormal, so msb_pos max value is 23
    UTy tmp = x_fra / y_fra;
    UTy fra_rem = x_fra - y_fra * tmp;
    int msb_pos = get_msb_pos(tmp);
    int tmp2 = exp_diff + msb_pos;
    if (tmp2 > __iml_fp_config<Ty>::bias)
      return __handling_fp_overflow<Ty>(z_sig, rd);

    if (tmp2 >= (1 - __iml_fp_config<Ty>::bias)) {
      // Fall into normal floating point range
      z_exp = tmp2 + __iml_fp_config<Ty>::bias;
      // For fp32, starting msb_pos bits in fra comes from tmp and we need
      // 23 - msb_pos( + grs) more bits from fraction division.
      z_fra = ((one_bits << msb_pos) - 1) & tmp;
      z_fra = z_fra << ((std::numeric_limits<Ty>::digits - 1) - msb_pos);
      UTy fra_bits_quo = fra_uint_div(
          fra_rem, y_fra, std::numeric_limits<Ty>::digits - msb_pos + 2);
      z_fra = z_fra | (fra_bits_quo >> 3);
      int rb = __handling_rounding(z_sig, z_fra, fra_bits_quo & 0x7, rd);
      if (rb != 0) {
        z_fra++;
        if (z_fra > __iml_fp_config<Ty>::fra_mask) {
          z_exp++;
          if (z_exp == __iml_fp_config<Ty>::exp_mask)
            return __handling_fp_overflow<Ty>(z_sig, rd);
        }
      }
      return __builtin_bit_cast(
          Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                  (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
    }

    // orignal value can be represented as (0.1xxxx.... * 2^tmp2)
    // which is equivalent to 0.00000...1xxxxx * 2^(-126)
    tmp2 = tmp2 + 1;
    if ((tmp2 + std::numeric_limits<Ty>::digits - 1) <=
        (1 - __iml_fp_config<Ty>::bias)) {
      bool above_half = false;
      if ((tmp2 + std::numeric_limits<Ty>::digits - 1) ==
          (1 - __iml_fp_config<Ty>::bias))
        above_half =
            !((x_fra == y_fra * tmp) && (tmp == (one_bits << msb_pos)));
      return __handling_fp_underflow<Ty, UTy>(z_sig, rd, above_half);
    } else {
      int rb;
      // Fall into subnormal floating point range. For fp32, there are -126 -
      // tmp2 leading zeros in final fra and we need get 23 + 126 + tmp2( + grs)
      // bits from fraction division.
      if (msb_pos >= (std::numeric_limits<Ty>::digits +
                      __iml_fp_config<Ty>::bias + tmp2)) {
        unsigned fra_discard_bits = msb_pos + 3 - __iml_fp_config<Ty>::bias -
                                    std::numeric_limits<Ty>::digits - tmp2;
        z_fra = tmp >> fra_discard_bits;
        int grs_bits = (tmp >> (fra_discard_bits - 3)) & 0x7;
        if ((grs_bits & 0x1) == 0x0) {
          if ((tmp & ((0x1 << (fra_discard_bits - 3)) - 0x1)) || (fra_rem != 0))
            grs_bits = grs_bits | 0x1;
        }
        rb = __handling_rounding(z_sig, z_fra, grs_bits, rd);
      } else {
        // For fp32, we need to get (23 + 126 + tmp2 + 3) - (msb_pos + 1) bits
        // from fra division and the last bit is sticky bit.
        z_fra = tmp;
        unsigned fra_get_bits = std::numeric_limits<Ty>::digits +
                                __iml_fp_config<Ty>::bias + tmp2 - msb_pos;
        z_fra = z_fra << fra_get_bits;
        UTy fra_bits_quo = fra_uint_div(fra_rem, y_fra, fra_get_bits);
        z_fra = z_fra | fra_bits_quo;
        int grs_bits = z_fra & 0x7;
        z_fra = z_fra >> 3;
        rb = __handling_rounding(z_sig, z_fra, grs_bits, rd);
      }
      if (rb != 0) {
        z_fra++;
        if (z_fra > __iml_fp_config<Ty>::fra_mask) {
          z_exp++;
          z_fra = 0x0;
        }
      }
      return __builtin_bit_cast(
          Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                  (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
    }
  } else {
    // x_fra < y_fra, the final result can be represented as
    // (2^exp_diff) * 0.000...01xxxxx
    unsigned lz = 0;
    UTy x_tmp = x_fra;
    x_tmp = x_tmp << 1;
    while (x_tmp < y_fra) {
      lz++;
      x_tmp = x_tmp << 1;
    }
    // x_fra < y_fra, the final result can be represented as
    // (2^exp_diff) * 0.000...01xxxxx... which is equivalent to
    // 2 ^ (exp_diff - lz - 1) * 1.xxxxx...
    int nor_exp = exp_diff - lz - 1;
    if (nor_exp > __iml_fp_config<Ty>::bias)
      return __handling_fp_overflow<Ty>(z_sig, rd);

    if (nor_exp >= (1 - __iml_fp_config<Ty>::bias)) {
      z_exp = nor_exp + __iml_fp_config<Ty>::bias;
      x_fra = x_fra << lz;
      UTy fra_bits_quo =
          fra_uint_div(x_fra, y_fra, 3 + std::numeric_limits<Ty>::digits);
      z_fra = (fra_bits_quo >> 3) & __iml_fp_config<Ty>::fra_mask;
      int grs_bits = fra_bits_quo & 0x7;
      int rb = __handling_rounding(z_sig, z_fra, grs_bits, rd);
      if (rb != 0x0) {
        z_fra++;
        if (z_fra > __iml_fp_config<Ty>::fra_mask) {
          z_exp++;
          z_fra = 0x0;
          if (z_exp == __iml_fp_config<Ty>::exp_mask)
            return __handling_fp_overflow<Ty>(z_sig, rd);
        }
      }
      return __builtin_bit_cast(
          Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                  (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
    }

    // Fall into subnormal range or underflow happens. For fp32,
    // nor_exp < -126, so (-126 - exp_diff + lz + 1) > 0 which means
    // (lz - exp_diff - 126) >= 0
    unsigned lzs = lz - __iml_fp_config<Ty>::bias - exp_diff + 1;
    if (lzs >= (std::numeric_limits<Ty>::digits - 1)) {
      bool above_half = false;
      if (lzs == (std::numeric_limits<Ty>::digits - 1)) {
        if ((x_fra << (lz + 1)) > y_fra)
          above_half = true;
      }
      return __handling_fp_underflow<Ty>(z_sig, rd, above_half);
    } else {
      x_fra = x_fra << lz;
      UTy fra_bits_quo =
          fra_uint_div(x_fra, y_fra, std::numeric_limits<Ty>::digits - lzs + 2);
      z_fra = fra_bits_quo >> 3;
      int grs_bits = fra_bits_quo & 0x7;
      int rb = __handling_rounding(z_sig, z_fra, grs_bits, rd);
      if (rb != 0x0) {
        z_fra++;
        if (z_fra > __iml_fp_config<Ty>::fra_mask) {
          z_exp++;
          z_fra = 0x0;
        }
      }
      return __builtin_bit_cast(
          Ty, (z_sig << (sizeof(Ty) * 8 - 1)) |
                  (z_exp << (std::numeric_limits<Ty>::digits - 1)) | z_fra);
    }
  }
}

unsigned get_grs_bits(uint64_t dbits, unsigned bit_num) {
  if (bit_num == 1)
    return (dbits & 0x1) << 2;
  else if (bit_num == 2)
    return (dbits & 0x3) << 1;
  else {
    uint64_t Bit1 = 1;
    unsigned grs = dbits >> (bit_num - 3);
    unsigned sbit_or = dbits & ((Bit1 << (bit_num - 3)) - 1);
    return (sbit_or == 0) ? grs : (grs | 0x1);
  }
}

unsigned get_grs_bits(__iml_ui128 dbits, unsigned bit_num) {
  if (bit_num == 1)
    return static_cast<uint32_t>(dbits & 0x1) << 2;
  else if (bit_num == 2)
    return static_cast<uint32_t>(dbits & 0x3) << 1;
  else {
    __iml_ui128 Bit1(1);
    unsigned grs = static_cast<unsigned>(dbits >> (bit_num - 3));
    unsigned sbit_or =
        static_cast<unsigned>(dbits & ((Bit1 << (bit_num - 3)) - 1));
    return (sbit_or == 0) ? grs : (grs | 0x1);
  }
}

template <typename FTy, typename UTy, typename DSUTy>
FTy __fma_helper_ss(int x_exp, DSUTy x_fra_ds, int y_exp, DSUTy y_fra_ds,
                    UTy sig, int rd) {
  size_t nshifts = x_exp - y_exp, lz1 = 0, msb_pos1, msb_pos2;
  UTy r_fra;
  DSUTy discarded_bits(0), Bit1(1);

  if constexpr (std::is_same<DSUTy, __iml_ui128>::value)
    msb_pos1 = y_fra_ds.ui128_msb_pos();
  else
    msb_pos1 = get_msb_pos(y_fra_ds);

  if (nshifts <= msb_pos1) {
    x_fra_ds += (y_fra_ds >> nshifts);
    discarded_bits = y_fra_ds & ((Bit1 << nshifts) - 1);
  } else {
    lz1 = nshifts - (1 + msb_pos1);
    discarded_bits = y_fra_ds;
  }

  if constexpr (std::is_same<DSUTy, __iml_ui128>::value)
    msb_pos2 = x_fra_ds.ui128_msb_pos();
  else
    msb_pos2 = get_msb_pos(x_fra_ds);

  // Result without rounding can be represented as:
  // 2^x_exp * x_fra_ds which is equivalent to:
  // 2^x_exp * 1.dd...d * 2^(msb_pos2) which is:
  // 2^temp * 1.dd...d
  // temp >= x_exp >= -149 for fp32 and
  // temp >= x_exp >= -1074 for fp64
  int temp = x_exp + msb_pos2;
  if (temp > __iml_fp_config<FTy>::bias)
    return __handling_fp_overflow<FTy>(sig, rd);

  constexpr int fra_digits = std::numeric_limits<FTy>::digits - 1;
  uint32_t g_bit = 0, r_bit = 0, s_bit = 0, grs_bits = 0, rb;

  if (msb_pos2 > fra_digits) {
    int dbits = msb_pos2 - fra_digits;
    r_fra = static_cast<UTy>((x_fra_ds & ((Bit1 << msb_pos2) - 1)) >> dbits);
    if (dbits == 1) {
      g_bit = static_cast<uint32_t>(x_fra_ds & 0x1);
      if (lz1 == 0) {
        if (nshifts == 1) {
          r_bit = static_cast<uint32_t>(discarded_bits & 0x1);
        } else {
          DSUTy t1 = (discarded_bits & (Bit1 << (nshifts - 1)));
          r_bit = static_cast<uint32_t>(t1 >> (nshifts - 1));
          if ((((Bit1 << (nshifts - 1)) - 1) & discarded_bits) != 0)
            s_bit = 1;
        }
      } else {
        r_bit = 0;
        if (discarded_bits != 0)
          s_bit = 1;
      }
      grs_bits = (g_bit << 2) | (r_bit << 1) | s_bit;
    } else if (dbits == 2) {
      g_bit = static_cast<uint32_t>((x_fra_ds & 0x2)) >> 1;
      r_bit = static_cast<uint32_t>(x_fra_ds & 0x1);
      if (discarded_bits != 0)
        s_bit = 1;
      grs_bits = (g_bit << 2) | (r_bit << 1) | s_bit;
    } else {
      grs_bits = static_cast<uint32_t>((x_fra_ds & ((Bit1 << dbits) - 1)) >>
                                       (dbits - 3));
      if ((grs_bits & 0x1) == 0x0) {
        auto t1 = ((Bit1 << (dbits - 3)) - 1) & x_fra_ds;
        if ((t1 != 0) || (discarded_bits != 0))
          s_bit = 1;
      }
      grs_bits = grs_bits | s_bit;
    }
  } else {

    int dbits = fra_digits - msb_pos2;
    r_fra = static_cast<UTy>(x_fra_ds & ((Bit1 << msb_pos2) - 1));
    r_fra <<= dbits;
    if (dbits >= nshifts) {
      r_fra |= static_cast<UTy>(discarded_bits << (dbits - nshifts));
    } else {
      if (lz1 > dbits) {
        // In this case, g_bit is 0, so we set grs_bits = 1 for simplification.
        grs_bits = 0x1;
      } else if (lz1 == dbits) {
        // In this case, g_bit is 1.
        if (lz1) {
          if (discarded_bits == (Bit1 << (nshifts - lz1 - 1)))
            grs_bits = 4;
          else
            grs_bits = 5;
        } else {
          if (nshifts == 0x0)
            grs_bits = 0x0;
          else if (nshifts == 0x1)
            grs_bits = static_cast<uint32_t>(discarded_bits & 0x1) << 2;
          else if (nshifts == 0x2)
            grs_bits = static_cast<uint32_t>(discarded_bits & 0x3) << 1;
          else {
            grs_bits = static_cast<uint32_t>(discarded_bits >> (nshifts - 3));
            if ((discarded_bits & ((Bit1 << (nshifts - 3)) - 1)) != 0)
              grs_bits = grs_bits | 1;
          }
        }
      } else {
        r_fra |= static_cast<UTy>(discarded_bits >> (nshifts - dbits));
        if ((nshifts - dbits) == 1) {
          grs_bits = static_cast<uint32_t>(discarded_bits & 0x1) << 2;
        } else if ((nshifts - dbits) == 2) {
          grs_bits = static_cast<uint32_t>(discarded_bits & 0x3) << 1;
        } else {

          if ((discarded_bits & (Bit1 << (nshifts - dbits - 1))) != 0)
            g_bit = 4;
          if ((discarded_bits & (Bit1 << (nshifts - dbits - 2))) != 0)
            r_bit = 2;
          if ((discarded_bits & ((Bit1 << (nshifts - dbits - 2)) - 1)) != 0)
            s_bit = 1;
          grs_bits = g_bit | r_bit | s_bit;
        }
      }
    }
  }
  rb = __handling_rounding(sig, r_fra, grs_bits, rd);
  r_fra += rb;
  UTy r_exp = temp + __iml_fp_config<FTy>::bias;
  if (r_fra > __iml_fp_config<FTy>::fra_mask) {
    r_fra = 0;
    r_exp++;
  }
  if (r_exp == __iml_fp_config<FTy>::exp_mask)
    return __handling_fp_overflow<FTy>(sig, rd);
  return __builtin_bit_cast(FTy, (sig << (sizeof(FTy) * 8 - 1)) |
                                     (r_exp << fra_digits) | r_fra);
}

template <typename FTy, typename UTy, typename DSUTy>
FTy __fma_helper_ds(int x_exp, DSUTy x_fra_ds, int y_exp, DSUTy y_fra_ds,
                    UTy x_sig, int rd) {
  size_t nshifts = x_exp - y_exp;
  UTy r_fra, r_exp;
  DSUTy discarded_bits(0), Bit1(1);
  uint32_t grs_bit_num = nshifts, grs_bits = 0, rb = 0;
  size_t lo = 0, msb_pos1;
  if constexpr (std::is_same<DSUTy, __iml_ui128>::value)
    msb_pos1 = y_fra_ds.ui128_msb_pos();
  else
    msb_pos1 = get_msb_pos(y_fra_ds);

  if (nshifts <= msb_pos1) {
    discarded_bits = y_fra_ds & ((Bit1 << nshifts) - 1);
    y_fra_ds = y_fra_ds >> nshifts;
    if (discarded_bits != 0) {
      if (x_fra_ds > y_fra_ds) {
        x_fra_ds -= 1;
        x_fra_ds -= y_fra_ds;
        discarded_bits = ~discarded_bits + 1;
        discarded_bits &= ((Bit1 << nshifts) - 1);
      } else if (x_fra_ds == y_fra_ds) {
        x_sig = x_sig ^ 1;
        x_fra_ds = 0;
      } else {
        x_sig = x_sig ^ 1;
        x_fra_ds = y_fra_ds - x_fra_ds;
      }
    } else {
      if (x_fra_ds > y_fra_ds) {
        x_fra_ds -= y_fra_ds;
      } else if (x_fra_ds == y_fra_ds) {
        return 0;
      } else {
        x_fra_ds = y_fra_ds - x_fra_ds;
        x_sig = x_sig ^ 1;
      }
    }
  } else {
    // In this case, final result can't be subnormal value
    x_fra_ds -= 1;
    lo = nshifts - (1 + msb_pos1);
    discarded_bits = ~y_fra_ds + 1;
    discarded_bits &= ((Bit1 << (1 + msb_pos1)) - 1);
  }

  size_t msb_pos2 = 0;
  int temp = 0;
  if (x_fra_ds == 0) {
    // In this case, discarded_bits is non-zero otherwise 0. is returned.
    // Result without rounding can be represented as:
    // 2^x_exp * 0.00..01dd..d which is equivalent to:
    // 2^x_exp * 1.dd...d * 2^(-d_lz - 1) which is:
    // 2^temp * 1.dd...d
    int d_lz = get_leading_zeros_from(discarded_bits, nshifts);
    temp = x_exp - d_lz - 1;
    x_fra_ds = discarded_bits;
    discarded_bits = 0;
    msb_pos2 = nshifts - d_lz - 1;
  } else {
    // Result without rounding can be represented as:
    // 2^x_exp * x_fra_ds which is equivalent to:
    // 2^x_exp * 1.dd...d * 2^(msb_pos2) which is:
    // 2^temp * 1.dd...d
    if constexpr (std::is_same<DSUTy, __iml_ui128>::value)
      msb_pos2 = x_fra_ds.ui128_msb_pos();
    else
      msb_pos2 = get_msb_pos(x_fra_ds);
    temp = x_exp + msb_pos2;
  }

  // Result without rounding can be represented as:
  // 2^x_exp * x_fra_ds which is equivalent to:
  // 2^x_exp * 1.dd...d * 2^(msb_pos2) which is:
  // 2^temp * 1.dd...d
  if (temp > __iml_fp_config<FTy>::bias)
    return __handling_fp_overflow<FTy>(x_sig, rd);

  constexpr int fra_digits = std::numeric_limits<FTy>::digits - 1;
  if (temp >= (1 - __iml_fp_config<FTy>::bias)) {
    x_fra_ds &= ((Bit1 << msb_pos2) - 1);
    if (msb_pos2 > fra_digits) {
      DSUTy t_fra = x_fra_ds & ((Bit1 << (msb_pos2 - fra_digits)) - 1);
      x_fra_ds = x_fra_ds >> (msb_pos2 - fra_digits);
      if (msb_pos2 >= (fra_digits + 3)) {
        if (discarded_bits > 0)
          t_fra = t_fra | 0x1;
        grs_bit_num = msb_pos2 - fra_digits;
        discarded_bits = t_fra;
      } else if (msb_pos2 == (fra_digits + 2)) {
        t_fra = t_fra << 1;
        if (discarded_bits > 0)
          t_fra = t_fra | 1;
        discarded_bits = t_fra;
        grs_bit_num = 3;
      } else {
        t_fra = t_fra << 2;
        if (lo > 0)
          t_fra = t_fra | 2;
        else {
          DSUTy tp = discarded_bits & (Bit1 << (nshifts - 1));
          if ((nshifts > 0) && (tp != 0))
            t_fra = t_fra | 2;
        }

        if (lo > 1)
          t_fra = t_fra | 1;
        else {
          if ((lo == 1) && (discarded_bits != 0))
            t_fra = t_fra | 1;
          if ((lo == 0) && (nshifts > 0)) {
            if ((discarded_bits & ((Bit1 << (nshifts - 1)) - 1)) != 0)
              t_fra = t_fra | 1;
          }
        }
        discarded_bits = t_fra;
        grs_bit_num = 3;
      }
    } else {
      x_fra_ds = x_fra_ds << (fra_digits - msb_pos2);
      if (nshifts >= (fra_digits - msb_pos2)) {
        // Need to fill 23 - msb_pos2 bit from pre-stored 'discarded_bits' to
        // final frac bits for fp32 and fill 52 - msb_pos2 for fp64
        if (lo == 0) {
          grs_bit_num = nshifts - (fra_digits - msb_pos2);
          x_fra_ds |= (discarded_bits >> grs_bit_num);
          discarded_bits &= ((Bit1 << grs_bit_num) - 1);
        } else {
          x_fra_ds |= ((Bit1 << (fra_digits - msb_pos2)) - 1);

          if (lo == (fra_digits - msb_pos2)) {
            grs_bit_num = nshifts - lo;
          } else {
            // In this case, g bit must be 1 and discarded_bits must be > 4, set
            // it to 5 for simplification.
            discarded_bits = 0x5;
            grs_bit_num = 3;
          }
        }
      } else {
        discarded_bits = discarded_bits << (fra_digits - msb_pos2 - nshifts);
        x_fra_ds |= discarded_bits;
        discarded_bits = 0;
      }
    }
    grs_bits = get_grs_bits(discarded_bits, grs_bit_num);
    r_exp = temp + __iml_fp_config<FTy>::bias;
    r_fra = static_cast<UTy>(x_fra_ds) & __iml_fp_config<FTy>::fra_mask;
  } else {
    // 2^x_exp * x_fra_ds which is equivalent to:
    // 2^x_exp * 1.dd...d * 2^(msb_pos2) which is:
    // 2^temp * 1.dd...d, final result falls into subnormal
    // range when temp < -126, so we need to normalize
    // it to: 2^-126 * 0.00...01dd...d for fp32 and for fp64,
    // fianl result falls into subnormal when temp < -1022, so
    // we need to normalize it to 2^-1022 * 0.00...01dd...d
    size_t sshifts = (1 - __iml_fp_config<FTy>::bias) - temp;
    size_t fra_bits_num = sshifts + msb_pos2;
    r_exp = 0;
    DSUTy t;
    if (fra_bits_num >= (fra_digits + 3)) {
      t = (Bit1 << (fra_bits_num - fra_digits)) - 1;
      t = (t & x_fra_ds) >> (fra_bits_num - (fra_digits + 3));
      grs_bits = static_cast<uint32_t>(t);
      t = x_fra_ds >> (fra_bits_num - fra_digits);
      r_fra = static_cast<UTy>(t);
      if (discarded_bits > 0)
        grs_bits = grs_bits | 1;
    } else if (fra_bits_num == (fra_digits + 1)) {
      t = (x_fra_ds & 1) << 2;
      grs_bits = static_cast<uint32_t>(t);
      r_fra = static_cast<UTy>(x_fra_ds >> 1);
      if ((nshifts > 0) && ((discarded_bits & (Bit1 << (nshifts - 1))) != 0))
        grs_bits |= 2;
      if (nshifts > 0) {
        discarded_bits -= discarded_bits & (Bit1 << (nshifts - 1));
        if (discarded_bits > 0)
          grs_bits |= 1;
      }
    } else if (fra_bits_num == (fra_digits + 2)) {
      grs_bits = static_cast<uint32_t>((x_fra_ds & 3) << 2);
      r_fra = static_cast<UTy>(x_fra_ds >> 2);
      if (discarded_bits > 0)
        grs_bits |= 1;
    } else {
      // fra_bits_num <= 23, need to fill 23 - fra_bits_num bits
      // from discarded_bits into final r_fra for fp32 and fill
      // 52 - fra_bits_num bits for fp64.
      size_t bits_fill_num = fra_digits - fra_bits_num;
      r_fra = static_cast<UTy>(x_fra_ds << bits_fill_num);
      if (nshifts && (discarded_bits > 0)) {
        if (bits_fill_num <= nshifts) {
          r_fra |=
              static_cast<UTy>(discarded_bits >> (nshifts - bits_fill_num));
          discarded_bits &= (Bit1 << (nshifts - bits_fill_num)) - 1;
          grs_bits = get_grs_bits(discarded_bits, (nshifts - bits_fill_num));
        } else
          r_fra |=
              static_cast<UTy>(discarded_bits << (bits_fill_num - nshifts));
      }
    }
  }

  rb = __handling_rounding(x_sig, r_fra, grs_bits, rd);
  r_fra += rb;
  if (r_fra > __iml_fp_config<FTy>::fra_mask) {
    r_fra = 0x0;
    r_exp++;
  }
  return __builtin_bit_cast(FTy, (x_sig << (sizeof(FTy) * 8 - 1)) |
                                     (r_exp << fra_digits) | r_fra);
}

template <typename FTy> FTy __fp_fma(FTy x, FTy y, FTy z, int rd) {
  typedef typename __iml_fp_config<FTy>::utype UTy;
  typedef typename __iml_get_double_size_unsigned<UTy>::utype DSUTy;
  constexpr int fra_digits = std::numeric_limits<FTy>::digits - 1;
  UTy x_bit = __builtin_bit_cast(UTy, x);
  UTy y_bit = __builtin_bit_cast(UTy, y);
  UTy z_bit = __builtin_bit_cast(UTy, z);
  UTy x_exp = (x_bit & __iml_fp_config<FTy>::pos_inf_bits) >> fra_digits;
  UTy y_exp = (y_bit & __iml_fp_config<FTy>::pos_inf_bits) >> fra_digits;
  UTy z_exp = (z_bit & __iml_fp_config<FTy>::pos_inf_bits) >> fra_digits;
  UTy x_fra = x_bit & __iml_fp_config<FTy>::fra_mask;
  UTy y_fra = y_bit & __iml_fp_config<FTy>::fra_mask;
  UTy z_fra = z_bit & __iml_fp_config<FTy>::fra_mask;
  UTy x_sig = x_bit >> (sizeof(FTy) * 8 - 1);
  UTy y_sig = y_bit >> (sizeof(FTy) * 8 - 1);
  UTy z_sig = z_bit >> (sizeof(FTy) * 8 - 1);
  UTy xy_sig = x_sig ^ y_sig;
  DSUTy Bit1(1);
  constexpr UTy NAN_BITS = __iml_fp_config<FTy>::nan_bits;
  constexpr UTy INF_BITS = __iml_fp_config<FTy>::pos_inf_bits;
  unsigned is_sig_diff = xy_sig ^ z_sig;

  if ((x_exp == __iml_fp_config<FTy>::exp_mask) && (x_fra != 0x0))
    return __builtin_bit_cast(FTy, NAN_BITS);

  if ((y_exp == __iml_fp_config<FTy>::exp_mask) && (y_fra != 0x0))
    return __builtin_bit_cast(FTy, NAN_BITS);

  if ((z_exp == __iml_fp_config<FTy>::exp_mask) && (z_fra != 0x0))
    return __builtin_bit_cast(FTy, NAN_BITS);

  if ((x_exp == 0x0) && (x_fra == 0x0)) {
    if ((y_exp == __iml_fp_config<FTy>::exp_mask) && (y_fra == 0x0))
      return __builtin_bit_cast(FTy, NAN_BITS);
    else
      return z;
  }

  if ((y_exp == 0x0) && (y_fra == 0x0)) {
    if ((x_exp == __iml_fp_config<FTy>::exp_mask) && (x_fra == 0x0))
      return __builtin_bit_cast(FTy, NAN_BITS);
    else
      return z;
  }

  if (((x_exp == __iml_fp_config<FTy>::exp_mask) && (x_fra == 0x0)) ||
      ((y_exp == __iml_fp_config<FTy>::exp_mask) && (y_fra == 0x0))) {
    if ((z_exp == __iml_fp_config<FTy>::exp_mask) && (z_fra == 0x0))
      return is_sig_diff ? __builtin_bit_cast(FTy, NAN_BITS) : z;
    else
      return __builtin_bit_cast(FTy,
                                (INF_BITS | (xy_sig << (sizeof(FTy) * 8 - 1))));
  }

  if ((z_exp == 0x0) && (z_fra == 0x0))
    return __fp_mul(x, y, rd);

  int v_exp, w_exp;
  DSUTy x_fra_ds(x_fra), y_fra_ds(y_fra), w_fra_ds(z_fra), v_fra_ds;

  if (x_exp == 0x0)
    v_exp = 1 - __iml_fp_config<FTy>::bias;
  else {
    v_exp = static_cast<int>(x_exp) - __iml_fp_config<FTy>::bias;
    x_fra_ds = x_fra_ds | (Bit1 << fra_digits);
  }

  if (y_exp == 0x0)
    v_exp += 1 - __iml_fp_config<FTy>::bias;
  else {
    v_exp += static_cast<int>(y_exp) - __iml_fp_config<FTy>::bias;
    y_fra_ds = y_fra_ds | (Bit1 << fra_digits);
  }

  v_exp = v_exp - 2 * fra_digits;
  v_fra_ds = x_fra_ds * y_fra_ds;

  // The result of x * y can be represented as:
  // 2^(v_exp - 46) * (x_fra_ds * y_fra_ds) for fp32 and
  // 2^(v_exp - 104) * (x_fra_ds * y_fra_ds) for fp64
  // x_fra_ds * y_fra_ds is non-zero and can be represented as:
  // 1.dddd...d * 2^(msb_pos1), so x * y representation is:
  // 2^(v_exp - 46 + msb_pos1) * 1.dddd...d for fp32 and
  // 2^(v_exp - 104 + msb_pos1) * 1.dddd...d for fp64
  size_t msb_pos1;
  if constexpr (std::is_same<DSUTy, __iml_ui128>::value)
    msb_pos1 = v_fra_ds.ui128_msb_pos();
  else
    msb_pos1 = get_msb_pos(v_fra_ds);

  int temp = v_exp + msb_pos1;
  int is_mid = 0;
  if ((((Bit1 << msb_pos1) - 1) & v_fra_ds) == 0x0)
    is_mid = 1;

  // x * y overflows
  if (temp > __iml_fp_config<FTy>::bias) {
    if ((z_exp == __iml_fp_config<FTy>::exp_mask) && (z_fra == 0x0) &&
        is_sig_diff)
      return __builtin_bit_cast(FTy, NAN_BITS);

    if (!is_sig_diff || (temp > (1 + __iml_fp_config<FTy>::bias)))
      return __handling_fp_overflow<FTy>(xy_sig, rd);
  }

  if ((z_exp == __iml_fp_config<FTy>::exp_mask) && (z_fra == 0x0))
    return z;

  // x * y is too small and z is non-zero value, all mant bits of x * y are
  // dicarded
  if (temp < (1 - __iml_fp_config<FTy>::bias - fra_digits)) {
    UTy rb = 0;
    uint32_t grs_bit = 0;
    if (is_sig_diff) {
      if (z_fra == 0x0) {
        z_fra = __iml_fp_config<FTy>::fra_mask;
        // z_exp is non-zero otherwise, z is 0
        z_exp--;
      } else
        z_fra--;
      if ((temp == (-__iml_fp_config<FTy>::bias - fra_digits)) &&
          (z_exp <= 0x1))
        grs_bit = is_mid ? 0x4 : 0x1;
      else
        grs_bit = 0x5;
    } else {
      if ((temp == (-__iml_fp_config<FTy>::bias - fra_digits)) &&
          (z_exp <= 0x1)) {
        // In this case, g_bit is 1, if s_bit and r_bit aren't both 0, we
        // set grs_bit to be 0x5 for simplification.
        grs_bit = is_mid ? 0x4 : 0x5;
      } else {
        // In this case, g_bit is 0, r_bit and s_bit won't be both 0, so we
        // set grs to be 1 for simplification.
        grs_bit = 0x1;
      }
    }
    rb = __handling_rounding(z_sig, z_fra, grs_bit, rd);
    z_fra += rb;
    if (z_fra > __iml_fp_config<FTy>::fra_mask) {
      z_fra = 0x0;
      z_exp++;
    }
    if (z_exp == __iml_fp_config<FTy>::exp_mask)
      return __handling_fp_overflow<FTy>(z_sig, rd);
    return __builtin_bit_cast(FTy, (z_sig << (sizeof(FTy) * 8 - 1)) |
                                       (z_exp << fra_digits) | z_fra);
  }

  if (z_exp == 0x0)
    w_exp = 1 - __iml_fp_config<FTy>::bias;
  else {
    w_exp = static_cast<int>(z_exp) - __iml_fp_config<FTy>::bias;
    w_fra_ds = w_fra_ds | (Bit1 << fra_digits);
  }

  if (temp < (1 - __iml_fp_config<FTy>::bias)) {
    size_t nshifts = w_exp - temp;
    UTy rb = 0;
    uint32_t grs_bits = 0;
    // After normalization, x * y can be represented as:
    // 2^(w_exp) * 0.00...01dd...d whose fraction has nshifts - 1 leading zeros
    // nshifts + msb_pos1 > 52
    size_t discarded_bits = nshifts + msb_pos1 - fra_digits;
    if (discarded_bits > (msb_pos1 + 1)) {
      if (!is_sig_diff) {
        // In this case, grs_bits < 4, we set it to 1 for simplification.
        grs_bits = 1;
      } else {
        if (z_fra == 0) {
          z_fra = __iml_fp_config<FTy>::fra_mask;
          z_exp--;
        } else
          z_fra--;
        // In this case, grs > 4, we set it to 5 for simplification.
        grs_bits = 5;
      }
    } else if (discarded_bits == (msb_pos1 + 1)) {
      if (!is_sig_diff)
        grs_bits = is_mid ? 4 : 5;
      else {
        grs_bits = is_mid ? 4 : 1;
        if (z_fra == 0) {
          z_fra = __iml_fp_config<FTy>::fra_mask;
          z_exp--;
        } else
          z_fra--;
      }
    } else {
      if (!is_sig_diff) {
        z_fra += static_cast<UTy>(v_fra_ds >> discarded_bits);
        if (discarded_bits == 1)
          grs_bits = static_cast<uint32_t>((v_fra_ds & 1) << 2);
        else if (discarded_bits == 2)
          grs_bits = static_cast<uint32_t>((v_fra_ds & 3) << 1);
        else {
          grs_bits = static_cast<uint32_t>(
              (v_fra_ds & ((Bit1 << discarded_bits) - 1)) >>
              (discarded_bits - 3));
          if ((v_fra_ds & ((Bit1 << (discarded_bits - 3)) - 1)) != 0)
            grs_bits |= 1;
        }
        if (z_fra > __iml_fp_config<FTy>::fra_mask) {
          z_fra &= __iml_fp_config<FTy>::fra_mask;
          if (z_exp != 0) {
            uint32_t n_grs = (z_fra & 0x1) << 2;
            n_grs |= ((grs_bits & 4) >> 1);
            if (grs_bits & 3)
              n_grs = n_grs | 1;
            grs_bits = n_grs;
            z_fra = z_fra >> 1;
            z_exp++;
          } else
            z_exp = 1;
        }
      } else {
        UTy v_fra = static_cast<UTy>(v_fra_ds >> discarded_bits);
        DSUTy v_fra_dbits = v_fra_ds & ((Bit1 << discarded_bits) - 1);
        if ((z_exp == 0) && (z_fra <= v_fra)) {
          z_sig = xy_sig;
          z_fra = v_fra - z_fra;
          if (discarded_bits == 1)
            grs_bits = static_cast<uint32_t>((v_fra_ds & 0x1) << 2);
          else if (discarded_bits == 2)
            grs_bits = static_cast<uint32_t>((v_fra_ds & 0x3) << 1);
          else {
            grs_bits = static_cast<uint32_t>(
                (v_fra_ds & ((Bit1 << discarded_bits) - 1)) >>
                (discarded_bits - 3));
            uint32_t sbit_or =
                ((v_fra_ds & ((Bit1 << (discarded_bits - 3)) - 1)) != 0) ? 1
                                                                         : 0;
            grs_bits |= sbit_or;
          }
        } else {
          if (v_fra_dbits == 0) {
            if (z_fra > v_fra)
              z_fra = z_fra - v_fra;
            else {
              z_fra = z_fra + 1 + __iml_fp_config<FTy>::fra_mask;
              z_fra = z_fra - v_fra;
              unsigned t_lz = get_leading_zeros_from(z_fra, fra_digits);
              if (z_exp >= 1 + t_lz + 1) {
                z_exp -= t_lz + 1;
                z_fra = (z_fra << (t_lz + 1)) & __iml_fp_config<FTy>::fra_mask;
              } else {
                z_fra = z_fra << (z_exp - 1);
                z_exp = 0;
              }
            }
          } else {
            int need_norm = 0;
            v_fra_dbits = ~v_fra_dbits + 1;
            v_fra_dbits &= (Bit1 << discarded_bits) - 1;
            if (z_fra == 0x0) {
              z_fra = __iml_fp_config<FTy>::fra_mask - v_fra;
              need_norm = 1;
            } else {
              z_fra--;
              if (z_fra > v_fra)
                z_fra = z_fra - v_fra;
              else {
                z_fra = z_fra + 1 + __iml_fp_config<FTy>::fra_mask;
                z_fra = z_fra - v_fra;
                need_norm = 1;
              }
            }
            if (need_norm) {
              unsigned t_lz = get_leading_zeros_from(z_fra, fra_digits);
              unsigned t_fra_bits = 0;
              if (z_exp >= 1 + t_lz + 1) {
                z_exp -= t_lz + 1;
                z_fra = (z_fra << (t_lz + 1)) & __iml_fp_config<FTy>::fra_mask;
                t_fra_bits = t_lz + 1;
              } else {
                z_fra = z_fra << (z_exp - 1);
                z_exp = 0;
                t_fra_bits = 0;
              }
              if (t_fra_bits >= discarded_bits) {
                v_fra_dbits = v_fra_dbits << (t_fra_bits - discarded_bits);
                z_fra = z_fra | static_cast<UTy>(v_fra_dbits);
                v_fra_dbits = 0;
                // In this case, grs is 0, all discarded bits are left shifted
                // to be included in final frac.
              } else {
                z_fra = z_fra | static_cast<UTy>(v_fra_dbits >>
                                                 (discarded_bits - t_fra_bits));
                v_fra_dbits =
                    v_fra_dbits & ((Bit1 << (discarded_bits - t_fra_bits)) - 1);
                discarded_bits = discarded_bits - t_fra_bits;
              }
            }
            grs_bits = get_grs_bits(v_fra_dbits, discarded_bits);
          }
        }
      }
    }
    rb = __handling_rounding(z_sig, z_fra, grs_bits, rd);
    z_fra += rb;
    if (z_fra > __iml_fp_config<FTy>::fra_mask) {
      z_fra = 0;
      z_exp++;
    }
    if (z_exp == __iml_fp_config<FTy>::exp_mask)
      return __handling_fp_overflow<FTy>(z_sig, rd);
    return __builtin_bit_cast(FTy, (z_sig << (sizeof(FTy) * 8 - 1)) |
                                       (z_exp << fra_digits) | z_fra);
  }

  w_exp -= fra_digits;

  if (!is_sig_diff) {
    // x * y can be represented as:
    // 2^v_exp * v_fra_ds and z can be represented as:
    // 2^2_exp * w_fra_ds
    if (v_exp >= w_exp)
      return __fma_helper_ss<FTy, UTy, DSUTy>(v_exp, v_fra_ds, w_exp, w_fra_ds,
                                              z_sig, rd);
    else
      return __fma_helper_ss<FTy, UTy, DSUTy>(w_exp, w_fra_ds, v_exp, v_fra_ds,
                                              z_sig, rd);
  } else {
    if (v_exp >= w_exp)
      return __fma_helper_ds<FTy, UTy, DSUTy>(v_exp, v_fra_ds, w_exp, w_fra_ds,
                                              xy_sig, rd);
    else
      return __fma_helper_ds<FTy, UTy, DSUTy>(w_exp, w_fra_ds, v_exp, v_fra_ds,
                                              z_sig, rd);
  }
}

#endif
