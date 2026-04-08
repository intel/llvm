//==----------- types.hpp - sycl_ext_oneapi_fp8 ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/marray.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#ifdef __SYCL_DEVICE_ONLY__
// FP8 builtins
extern __DPCPP_SYCL_EXTERNAL
    uint8_t __builtin_spirv_ClampConvertFP16ToE4M3INTEL(sycl::half) noexcept;
extern __DPCPP_SYCL_EXTERNAL
    uint8_t __builtin_spirv_ConvertFP16ToE4M3EXT(sycl::half) noexcept;
extern __DPCPP_SYCL_EXTERNAL sycl::half
__builtin_spirv_ConvertE4M3ToFP16EXT(uint8_t) noexcept;
extern __DPCPP_SYCL_EXTERNAL
    uint8_t __builtin_spirv_ClampConvertBF16ToE4M3INTEL(
        sycl::ext::oneapi::bfloat16) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
    __builtin_spirv_ConvertBF16ToE4M3EXT(sycl::ext::oneapi::bfloat16) noexcept;
extern __DPCPP_SYCL_EXTERNAL sycl::ext::oneapi::bfloat16
__builtin_spirv_ConvertE4M3ToBF16EXT(uint8_t) noexcept;
extern __DPCPP_SYCL_EXTERNAL
    uint8_t __builtin_spirv_ClampConvertFP16ToE5M2INTEL(sycl::half) noexcept;
extern __DPCPP_SYCL_EXTERNAL
    uint8_t __builtin_spirv_ConvertFP16ToE5M2EXT(sycl::half) noexcept;
extern __DPCPP_SYCL_EXTERNAL sycl::half
__builtin_spirv_ConvertE5M2ToFP16EXT(uint8_t) noexcept;
extern __DPCPP_SYCL_EXTERNAL
    uint8_t __builtin_spirv_ClampConvertBF16ToE5M2INTEL(
        sycl::ext::oneapi::bfloat16) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
    __builtin_spirv_ConvertBF16ToE5M2EXT(sycl::ext::oneapi::bfloat16) noexcept;
extern __DPCPP_SYCL_EXTERNAL sycl::ext::oneapi::bfloat16
__builtin_spirv_ConvertE5M2ToBF16EXT(uint8_t) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
__builtin_spirv_ClampStochasticRoundFP16ToE5M2INTEL(sycl::half, uint32_t,
                                                    uint32_t *) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
__builtin_spirv_StochasticRoundFP16ToE5M2INTEL(sycl::half, uint32_t,
                                               uint32_t *) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
__builtin_spirv_ClampStochasticRoundBF16ToE5M2INTEL(sycl::ext::oneapi::bfloat16,
                                                    uint32_t,
                                                    uint32_t *) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
__builtin_spirv_StochasticRoundBF16ToE5M2INTEL(sycl::ext::oneapi::bfloat16,
                                               uint32_t, uint32_t *) noexcept;
#endif // __SYCL_DEVICE_ONLY__

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class saturation { none, finite };

enum class rounding {
  to_even,
  upward,
  toward_zero,
};

struct stochastic_seed {
  explicit stochastic_seed(uint32_t *pseed) : pseed(pseed) {}
  uint32_t *const pseed;
};

namespace detail {
static inline uint8_t RneClip(float x, uint8_t max) noexcept {
  float f = std::floor(x);
  float frac = x - f;
  uint8_t i = static_cast<uint8_t>(f);
  if (frac > 0.5f)
    ++i;
  else if (frac == 0.5f)
    i += (i & 1u); // ties to even
  return i > max ? max : i;
}

static inline uint8_t RoundClip(float x, uint8_t max, rounding R,
                                uint8_t sign_bit) noexcept {
  if (max == 0) {
    // No fraction bits (E8M0 path)
    if (R == rounding::upward) {
      // For sign-preserving formats, roundTowardPositive increments only for
      // positive values with a non-zero residual. Negative values stay at the
      // lower-magnitude encoding.
      if (!std::isnan(x) && sign_bit == 0u && x > 0.0f)
        return 1u;
      return 0u;
    }
    if (R == rounding::toward_zero || std::isnan(x))
      return 0u;
    if (x > 0.5f)
      return 1u;
    if (x == 0.5f)
      return 0u; // tie -> even (0)
    return 0u;
  }

  // Formats with fraction bits (E4M3, E5M2)
  if (R == rounding::upward) {
    if (sign_bit == 0u) {
      // Positive: ceil
      uint32_t ci = static_cast<uint32_t>(std::ceil(x));
      if (ci > max)
        ci = max;
      return static_cast<uint8_t>(ci);
    } else {
      // Negative: toward +inf => magnitude decreases -> floor
      uint32_t fi = static_cast<uint32_t>(std::floor(x));
      if (fi > max)
        fi = max;
      return static_cast<uint8_t>(fi);
    }
  }
  // default: round-to-nearest-even
  return RneClip(x, max);
}

static inline int BitWidth(uint32_t x) noexcept {
  int width = 0;
  while (x != 0u) {
    ++width;
    x >>= 1;
  }
  return width;
}

template <typename ToT> struct DirectBinary16Traits;

template <> struct DirectBinary16Traits<sycl::half> {
  static constexpr uint16_t SignMask = 0x8000u;
  static constexpr uint16_t FracMask = 0x03FFu;
  static constexpr uint16_t InfBits = 0x7C00u;
  static constexpr uint16_t MaxFiniteBits = 0x7BFFu;
  static constexpr uint16_t QuietNaNBits = 0x7E00u;
  static constexpr int FracBits = 10;
  static constexpr int Bias = 15;
  static constexpr int Emin = -14;
  static constexpr int Emax = 15;
};

template <> struct DirectBinary16Traits<sycl::ext::oneapi::bfloat16> {
  static constexpr uint16_t SignMask = 0x8000u;
  static constexpr uint16_t FracMask = 0x007Fu;
  static constexpr uint16_t InfBits = 0x7F80u;
  static constexpr uint16_t MaxFiniteBits = 0x7F7Fu;
  static constexpr uint16_t QuietNaNBits = 0x7FC0u;
  static constexpr int FracBits = 7;
  static constexpr int Bias = 127;
  static constexpr int Emin = -126;
  static constexpr int Emax = 127;
};

template <typename ToT> static inline ToT MakeDirectNaN() noexcept {
  if constexpr (std::is_same_v<ToT, sycl::half> ||
                std::is_same_v<ToT, sycl::ext::oneapi::bfloat16>) {
    return sycl::bit_cast<ToT>(DirectBinary16Traits<ToT>::QuietNaNBits);
  } else if constexpr (std::numeric_limits<ToT>::has_quiet_NaN) {
    return std::numeric_limits<ToT>::quiet_NaN();
  } else {
    return ToT{};
  }
}

template <typename ToT>
static inline ToT ConvertFloatToTarget(bool negative, uint32_t significand,
                                       int exp2, int srcFracBits,
                                       rounding R) noexcept {
  if (significand == 0u)
    return negative ? -ToT{0} : ToT{0};

  if constexpr (std::is_same_v<ToT, sycl::half> ||
                std::is_same_v<ToT, sycl::ext::oneapi::bfloat16>) {
    using Traits = DirectBinary16Traits<ToT>;
    const uint16_t sign = negative ? Traits::SignMask : 0u;
    const int sigBits = BitWidth(significand);
    const int unbiasedExp = exp2 + sigBits - 1 - srcFracBits;

    if (unbiasedExp > Traits::Emax) {
      return sycl::bit_cast<ToT>(static_cast<uint16_t>(
          sign | (R == rounding::toward_zero ? Traits::MaxFiniteBits
                                             : Traits::InfBits)));
    }

    if (unbiasedExp >= Traits::Emin) {
      const int shift = Traits::FracBits - (sigBits - 1);
      const uint32_t aligned = significand << shift;
      const uint16_t expField =
          static_cast<uint16_t>(unbiasedExp + Traits::Bias) << Traits::FracBits;
      const uint16_t fracField =
          static_cast<uint16_t>(aligned & Traits::FracMask);
      return sycl::bit_cast<ToT>(
          static_cast<uint16_t>(sign | expField | fracField));
    }

    const int subShift = exp2 - srcFracBits - Traits::Emin + Traits::FracBits;
    if (subShift < 0)
      return sycl::bit_cast<ToT>(sign);

    const uint32_t fracField = significand << subShift;
    if (fracField == 0u || fracField > Traits::FracMask)
      return sycl::bit_cast<ToT>(sign);

    return sycl::bit_cast<ToT>(
        static_cast<uint16_t>(sign | static_cast<uint16_t>(fracField)));
  } else if constexpr (std::is_floating_point_v<ToT>) {
    ToT magnitude =
        std::ldexp(static_cast<ToT>(significand), exp2 - srcFracBits);
    return negative ? -magnitude : magnitude;
  } else if constexpr (std::is_integral_v<ToT>) {
    const int shift = exp2 - srcFracBits;
    uint64_t magnitude = significand;
    if (shift >= 0)
      magnitude <<= shift;
    else if (-shift < 64)
      magnitude >>= -shift;
    else
      magnitude = 0u;

    if constexpr (std::is_signed_v<ToT>) {
      int64_t signedMagnitude = static_cast<int64_t>(magnitude);
      return static_cast<ToT>(negative ? -signedMagnitude : signedMagnitude);
    } else
      return static_cast<ToT>(magnitude);
  } else
    return ToT{};
}

template <int Ebits, int Mbits, typename ToT>
static inline ToT ConvertFromFP8_CPU(uint8_t b,
                                     rounding R = rounding::to_even) noexcept {
  static_assert((Ebits == 4 && Mbits == 3) || (Ebits == 5 && Mbits == 2) ||
                    (Ebits == 8 && Mbits == 0),
                "Unsupported FP8 (Ebits,Mbits) combination");

  constexpr int Bias = (1 << (Ebits - 1)) - 1;
  constexpr int Emin = 1 - Bias;
  constexpr uint8_t ExpMaskAll = static_cast<uint8_t>((1u << Ebits) - 1u);
  constexpr uint32_t FracDen = (Mbits == 0) ? 1u : (1u << Mbits);
  constexpr uint8_t MaxFrac = static_cast<uint8_t>(FracDen - 1u);

  // Extract fields.
  uint8_t sign_bit = (b & 0x80u) ? 1u : 0u;
  uint8_t frac = (Mbits == 0) ? 0u : static_cast<uint8_t>(b & MaxFrac);

  uint8_t exp = static_cast<uint8_t>((b >> Mbits) & ExpMaskAll);
  if constexpr (Ebits == 8 && Mbits == 0) {
    sign_bit = 0u;
    exp = b;
  }

  auto make_nan = [&]() -> ToT { return MakeDirectNaN<ToT>(); };

  // Handle exp = all ones (custom finite-only rules).
  if (exp == ExpMaskAll) {
    if constexpr (Ebits == 4 && Mbits == 3) {
      // E4M3: only frac==111 -> NaN, otherwise normal.
      if (frac == MaxFrac)
        return make_nan();
      // treat as normal finite
    } else if constexpr (Ebits == 5 && Mbits == 2) {
      // E5M2: NaN when frac in {01,10,11} i.e. frac != 00
      if (frac != 0)
        return make_nan();
      // frac==00 -> normal finite
    } else // E8M0: exp all ones -> NaN
      return make_nan();
  }

  // exp == 0 : zero or subnormal (if Mbits>0)
  if (exp == 0) {
    if constexpr (Mbits == 0) {
      // E8M0: exp==0 is the smallest normal (no subnormals)
      return ConvertFloatToTarget<ToT>(false, 1u, -Bias, 0, R);
    } else {
      if (frac == 0) {
        return ConvertFloatToTarget<ToT>(sign_bit != 0u, 0u, 0, 0, R);
      }
      // Subnormal: value = sign * (frac / 2^Mbits) * 2^(Emin)
      return ConvertFloatToTarget<ToT>(sign_bit != 0u, frac, Emin, Mbits, R);
    }
  }

  // Normal number.
  int E = static_cast<int>(exp) - Bias;
  const uint32_t significand =
      (Mbits == 0) ? 1u : (static_cast<uint32_t>(FracDen) + frac);
  return ConvertFloatToTarget<ToT>(sign_bit != 0u, significand, E, Mbits, R);
}

/// \brief Converts a given value to fp8 floating point with a rounding
/// mode to_even by default and saturation finite for host code.
/// \param h The input value to be converted.
/// \param R The rounding mode to be used during conversion.
/// \return uint8_t The converted 8-bit floating point value, MSB is sign bit,
/// Ebits bits exponent, Mbits bits mantissa.
template <int Ebits, int Mbits, typename T>
static inline uint8_t
ConvertToFP8_CPU(T h, rounding R = rounding::to_even) noexcept {
  // Specialized implementation for fp8_e8m0_x (Ebits=8, Mbits=0)
  if constexpr (Ebits == 8 && Mbits == 0) {
    // Format characteristics (finite-only, no zero, no infinity):
    //  - Bias: 127
    //  - Exponent field range used for normals: 0 .. 254  (E = ecode - 127 ->
    //  [-127, +127])
    //  - Encoding with exp==255 (0xFF) reserved for NaN (single payload 0xFF)
    //  - Value encoded when exponent field == 0:  +/- 2^{-127}
    //  - Max normal:  +/- 2^{127}  (~1.7014118e+38)
    //
    // Rounding mode: the public API restricts this format to rounding::upward.
    // Here we honor upward if passed; any other mode falls back to upward
    // behavior.
    //
    // Note: The format cannot represent zero; inputs with |x| < 2^{-127} map
    //       to the smallest magnitude normal with the input sign preserved
    //       (consistent with prior sign-preserving underflow behavior).
    //
    constexpr int Bias = 127;
    constexpr int Emin = -127;
    constexpr int Emax = 127;
    constexpr uint8_t NaNCode = 0xFF;                // 11111111
    constexpr uint8_t MaxExpField = 254;             // 255 reserved for NaN
    const float min_normal = std::ldexp(1.0f, Emin); // 2^{-127}
    const float max_normal = std::ldexp(1.0f, Emax); // 2^{127}

    float x = static_cast<float>(h);

    if (std::isnan(x))
      return NaNCode;

    uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
    float ax = std::fabs(x);

    // Handle underflow (|x| < min_normal) and x == 0: encode smallest normal
    // with sign.
    if (ax == 0.0f || ax < min_normal)
      return sign; // exp field = 0 -> E = -127

    // Handle overflow (|x| >= max_normal * (anything beyond representable)):
    if (ax >= max_normal)
      return static_cast<uint8_t>(sign | (MaxExpField)); // E = +127

    // Determine exponent E such that 2^E <= ax < 2^{E+1}
    int e2;
    float m = std::frexp(ax, &e2);
    int E = e2 - 1;

    // Upward rounding semantics:
    //  - For positive numbers: if not exact power-of-two, round up to next
    //  power (E+1) if within range.
    //  - For negative numbers: rounding toward +inf moves value toward zero, so
    //  keep current E.

    if (R == rounding::upward) {
      if (sign == 0x00) {
        // Round up (increase exponent) if possible.
        if (E < Emax)
          ++E;
        else
          E = Emax;
      }
    }

    // Clamp exponent just in case.
    if (E < Emin)
      E = Emin;
    if (E > Emax)
      E = Emax;

    uint8_t ecode = static_cast<uint8_t>(E + Bias); // 0 .. 254
    // ecode must never be 255 here.
    return static_cast<uint8_t>(sign | ecode);
  }

  constexpr int bias = (1 << (Ebits - 1)) - 1;
  // allow the top exponent field (ExpAllOnes) as a normal exponent except when
  // frac==MaxFrac (NaN)
  int emax = 0;
  int emin = 0;
  if constexpr (Ebits == 8)
    emax = 127;
  else {
    emax = (1 << Ebits) - 1 - bias; // ExpAllOnes - bias
    emin = 1 - bias;
  }
  constexpr uint8_t ExpAllOnes = static_cast<uint8_t>((1 << Ebits) - 1);
  constexpr uint8_t MaxFrac = static_cast<uint8_t>((1 << Mbits) - 1);
  constexpr uint8_t MaxFracForMaxNormal =
      (Ebits == 4 && Mbits == 3) || (Ebits == 5 && Mbits == 3)
          ? static_cast<uint8_t>(MaxFrac - 1u)
          : MaxFrac;
  constexpr uint8_t MaxExpForMaxNormal =
      (Ebits == 5 && Mbits == 2) ? static_cast<uint8_t>(ExpAllOnes - 1u)
                                 : ExpAllOnes;
  constexpr uint8_t MaxFracMask = MaxFrac;

  float x = static_cast<float>(h);
  uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
  if (std::isnan(x))
    return static_cast<uint8_t>(
        sign | ((ExpAllOnes << Mbits) | MaxFracMask)); // S.1111.111 -> NaN
  uint8_t sign_bit = sign ? 1u : 0u;
  float ax = std::fabs(x);
  const float max_finite =
      (2.0f - std::ldexp(1.0f, 1 - Mbits)) * std::ldexp(1.0f, emax);
  const float min_sub = std::ldexp(1.0f, emin - Mbits);

  if (ax > max_finite) {
    return static_cast<uint8_t>(
        sign | ((MaxExpForMaxNormal << Mbits) | MaxFracForMaxNormal));
  }

  if (ax < min_sub)
    return sign; // underflow

  int e2;
  float m = std::frexp(ax, &e2);
  int E = e2 - 1;

  if (E < emin) {
    float scaled = std::ldexp(ax, -emin) * static_cast<float>(1 << Mbits);
    uint32_t k = RoundClip(scaled, MaxFrac, R, sign_bit);
    if (k == 0)
      return sign;
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(k));
  }

  float y = m * 2.0f;
  float frac_scaled = (y - 1.0f) * static_cast<float>(1 << Mbits);
  uint32_t frac = RoundClip(frac_scaled, MaxFrac, R, sign_bit);
  if (frac == (1u << Mbits)) {
    frac = 0;
    ++E;
  }
  if (E > emax) {
    auto ret = static_cast<uint8_t>(
        sign | ((MaxExpForMaxNormal << Mbits) | MaxFracForMaxNormal));
    return ret;
  }
  uint8_t ecode = static_cast<uint8_t>(E + bias);
  auto ret = static_cast<uint8_t>(sign | (ecode << Mbits) |
                                  static_cast<uint8_t>(frac));
  return ret;
}

// Map E4M3 byte to integer
// then "nextUp" in that order, and map back.
// E4M3 finite-only: exp=0xF & frac!=0 => NaN (no Inf).
inline uint8_t nextE4M3(uint8_t b, bool up) {
  uint8_t exp = (b >> 3) & 0x0F;
  uint8_t frac = b & 0x07;
  // NaN -> NaN
  if (exp == 0x0F && frac)
    return b;
  uint8_t ord =
      (b & 0x80) ? static_cast<uint8_t>(~b) : static_cast<uint8_t>(b ^ 0x80);

  if (up) {
    if (ord == 0xFF)
      return b;
    ++ord;
  } else {
    if (ord == 0x00)
      return b;
    --ord;
  }
  return (ord & 0x80) ? static_cast<uint8_t>(ord ^ 0x80)
                      : static_cast<uint8_t>(~ord);
}

template <typename YiT, typename T>
uint8_t round(rounding r, uint8_t b, YiT yi, T vi) {
  switch (r) {
  case rounding::upward: {
    if (yi < vi)
      return nextE4M3(b, /*up=*/true);
    break;
  }
  case rounding::toward_zero:
    if (vi > 0.0f && yi > vi) {
      return nextE4M3(b, /*up=*/false);
    } else if (vi < 0.0f && yi < vi) {
      return nextE4M3(b, /*up=*/true);
    }
    break;
  default:
    break;
  }
  return b;
}

template <typename T>
static inline uint8_t ConvertToE8M0_CPU(T x, rounding R,
                                        saturation S) noexcept {
  // E8M0: unsigned 8-bit exponent code, bias 127.
  // Code 0xFF reserved for NaN. No Inf, no subnormals, no signed zero.
  constexpr int Bias = 127;
  constexpr int Emin = -127;
  constexpr int Emax = 127;
  constexpr uint8_t NaNCode = 0xFF;
  constexpr uint8_t MaxFiniteCode = 0xFE;

  // NaN and Inf checks only apply to non-integral types.
  if constexpr (!std::is_integral_v<T>) {
    if (std::isnan(static_cast<float>(x)))
      return NaNCode;
    if (std::isinf(static_cast<float>(x)))
      return (S == saturation::finite) ? MaxFiniteCode : NaNCode;
  }

  // Compute absolute value in the natural type T.
  T ax;
  if constexpr (std::is_unsigned_v<T>)
    ax = x;
  else if constexpr (std::is_signed_v<T> && std::is_integral_v<T>)
    ax = x < T(0) ? static_cast<T>(-x) : x;
  else
    ax = static_cast<T>(std::fabs(static_cast<float>(x)));

  // Zero check in natural type.
  if (ax == T(0))
    return 0x00;

  // Convert to float for frexp/ldexp-based exponent extraction.
  float fax = static_cast<float>(ax);

  // Underflow: map to min normal (code 0).
  // Min normal = 2^-127.
  const float min_normal = std::ldexp(1.0f, Emin);
  if (fax < min_normal)
    return 0x00;

  // Overflow and "too large": clamp or NaN depending on saturation.
  const float max_normal = std::ldexp(1.0f, Emax); // 2^127
  if (fax >= max_normal)
    return (S == saturation::finite) ? MaxFiniteCode : NaNCode;

  // Determine E such that 2^E <= fax < 2^(E+1).
  int e2 = 0;
  float m = std::frexp(fax, &e2); // fax = m * 2^e2, m in [0.5, 1)
  int E = e2 - 1;

  // With no mantissa, representables are exact powers of two.
  // Choose between 2^E and 2^(E+1) based on rounding mode.
  const bool is_exact_power_of_two = (m == 0.5f);

  switch (R) {
  case rounding::upward:
    // toward +inf; with no sign, this is "ceil in magnitude".
    if (!is_exact_power_of_two && E < Emax)
      ++E;
    break;
  case rounding::toward_zero:
    // toward -inf / toward 0: both pick the lower power for non-exact.
    break;
  case rounding::to_even:
  default: {
    if (!is_exact_power_of_two) {
      // Nearest of {2^E, 2^(E+1)} w/ ties-to-even (even exponent on tie).
      float lo = std::ldexp(1.0f, E);
      float hi = std::ldexp(1.0f, E + 1);
      float dlo = fax - lo;
      float dhi = hi - fax;
      if (dhi < dlo) {
        if (E < Emax)
          ++E;
      } else if (dhi == dlo) {
        // tie -> even exponent
        if ((E & 1) != 0 && E < Emax)
          ++E;
      }
    }
    break;
  }
  }

  if (E < Emin)
    E = Emin;
  if (E > Emax)
    E = Emax;

  uint8_t code = static_cast<uint8_t>(E + Bias); // 0..254
  return code;
}

template <typename ToT>
static inline ToT ConvertFromE8M0_CPU(uint8_t code, rounding R) noexcept {
  constexpr int Bias = 127;
  if (code == 0xFF) {
    return MakeDirectNaN<ToT>();
  }
  return ConvertFloatToTarget<ToT>(false, 1u, static_cast<int>(code) - Bias, 0,
                                   R);
}

} // namespace detail

template <size_t N> class fp8_e4m3_x {
  static constexpr size_t NExpBits = 4;
  static constexpr size_t NFracBits = 3;
  static constexpr float MaxNormal = 448.0f;
  static constexpr float MinSubnormal = 0.00000762939453125f; // 2^-17
  static constexpr uint8_t NaNCode = 0xFF;
  static constexpr uint8_t MaxFiniteCode =
      0x7E; // 0.1111.110 (positive max normal)

  static_assert(N == 1 || N == 2,
                "fp8_e4m3_x: Template argument N must be 1 or 2");

  template <typename T> uint8_t ConvertToFP8(T h) {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_spirv_ClampConvertFP16ToE4M3INTEL(h);
#else
    return detail::ConvertToFP8_CPU<4, 3, sycl::half>(h, rounding::to_even);
#endif
  }

  uint8_t ConvertBF16ToFP8(bfloat16 h) {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_spirv_ClampConvertBF16ToE4M3INTEL(h);
#else
    return detail::ConvertToFP8_CPU<4, 3, bfloat16>(h, rounding::to_even);
#endif
  }

  template <typename T> T ConvertFromFP8(uint8_t v) const {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::half hi = __builtin_spirv_ConvertE4M3ToFP16EXT(v);
    return static_cast<T>(hi);
#else
    return detail::ConvertFromFP8_CPU<4, 3, T>(v);
#endif
  }

  bfloat16 ConvertBF16FromFP8(uint8_t v) const {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_spirv_ConvertE4M3ToBF16EXT(v);
#else
    return detail::ConvertFromFP8_CPU<4, 3, bfloat16>(v);
#endif
  }

  void CheckConstraints(rounding r) const {
    if (r != rounding::to_even)
      throw std::invalid_argument(
          "fp8_e4m3_x: only rounding::to_even is supported");
  }

public:
  fp8_e4m3_x() = default;
  fp8_e4m3_x(const fp8_e4m3_x &) = default;

  ~fp8_e4m3_x() = default;
  fp8_e4m3_x &operator=(const fp8_e4m3_x &) = default;

  // Construct from pack of half, float, double.
  // Available only when the size of the pack is equal to N.

  template <typename... Types,
            typename = std::enable_if_t<
                (sizeof...(Types) == N) &&
                (((std::is_same_v<std::decay_t<Types>, half>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, bfloat16>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, float>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, double>) && ...))>>
  explicit fp8_e4m3_x(Types... v) {
    if constexpr (((std::is_same_v<std::decay_t<Types>, bfloat16>) && ...)) {
      const bfloat16 in[N] = {static_cast<bfloat16>(v)...};
      for (size_t i = 0; i < N; ++i)
        vals[i] = ConvertBF16ToFP8(in[i]);
      return;
    }
    const sycl::half in[N] = {v...};
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(in[i]);
  }

  // Construct from an array of half, bfloat16, float, double.
  explicit fp8_e4m3_x(sycl::half const (&v)[N],
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i]);
  }

  explicit fp8_e4m3_x(bfloat16 const (&v)[N], rounding r = rounding::to_even) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertBF16ToFP8(v[i]);
  }

  explicit fp8_e4m3_x(float const (&v)[N], rounding r = rounding::to_even) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i]);
  }

  explicit fp8_e4m3_x(double const (&v)[N]) {
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i]);
  }

  // Construct from an marray of half, bfloat16, float, double.
  explicit fp8_e4m3_x(const sycl::marray<sycl::half, N> &v,
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i]);
  }

  explicit fp8_e4m3_x(const sycl::marray<bfloat16, N> &v,
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertBF16ToFP8(v[i]);
  }

  explicit fp8_e4m3_x(const sycl::marray<float, N> &v,
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i]);
  }

  explicit fp8_e4m3_x(const sycl::marray<double, N> &v) {
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i]);
  }

  // Construct from integer types.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e4m3_x(short val) {
    vals[0] = ConvertToFP8(val);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e4m3_x(int val) {
    vals[0] = ConvertToFP8(val);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e4m3_x(long val) {
    vals[0] = ConvertToFP8(val);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e4m3_x(long long val) {
    vals[0] = ConvertToFP8(val);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e4m3_x(unsigned short val) {
    vals[0] = ConvertToFP8(val);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e4m3_x(unsigned int val) {
    vals[0] = ConvertToFP8(val);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e4m3_x(unsigned long val) {
    vals[0] = ConvertToFP8(val);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e4m3_x(unsigned long long val) {
    vals[0] = ConvertToFP8(val);
  }

  // Assign (operator) from half, bfloat16, float, double, and integer types.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(sycl::half val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(bfloat16 val) {
    vals[0] = ConvertBF16ToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(float val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(double val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(short val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(int val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(long val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(long long val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(unsigned short val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(unsigned int val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(unsigned long val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e4m3_x &operator=(unsigned long long val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }

  // Convert to half, bfloat16, float, double.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator half() const {
    return ConvertFromFP8<sycl::half>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bfloat16() const {
    return ConvertBF16FromFP8(vals[0]);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator float() const {
    return ConvertFromFP8<float>(vals[0]);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator double() const {
    return ConvertFromFP8<double>(vals[0]);
  }

  // Convert to integer types.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator char() const {
    return ConvertFromFP8<char>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator signed char() const {
    return ConvertFromFP8<signed char>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator short() const {
    return ConvertFromFP8<short>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator int() const {
    return ConvertFromFP8<int>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator long() const {
    return ConvertFromFP8<long>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator long long() const {
    return ConvertFromFP8<long long>(vals[0]);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned char() const {
    return ConvertFromFP8<unsigned char>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned short() const {
    return ConvertFromFP8<unsigned short>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned int() const {
    return ConvertFromFP8<unsigned int>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned long() const {
    return ConvertFromFP8<unsigned long>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned long long() const {
    return ConvertFromFP8<unsigned long long>(vals[0]);
  }

  // Convert to bool
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bool() const {
#ifdef __SYCL_DEVICE_ONLY__
    // detect +0 / -0
    sycl::half h = __builtin_spirv_ConvertE4M3ToFP16EXT(vals[0]);
    return h != 0;
#else
    // no need to convert, just check sign bit and 0s
    return vals[0] != 0 && vals[0] != 0x80;
#endif
  }

  // Convert to marray of half, bfloat16, float

  explicit operator sycl::marray<sycl::half, N>() const {
    sycl::marray<sycl::half, N> ret;
    for (size_t i = 0; i < N; ++i)
      ret[i] = ConvertFromFP8<sycl::half>(vals[i]);
    return ret;
  }

  explicit operator sycl::marray<bfloat16, N>() const {
    sycl::marray<bfloat16, N> ret;
    for (size_t i = 0; i < N; ++i)
      ret[i] = ConvertBF16FromFP8(vals[i]);
    return ret;
  }

  explicit operator sycl::marray<float, N>() const {
    sycl::marray<float, N> ret;
    for (size_t i = 0; i < N; ++i)
      ret[i] = ConvertFromFP8<float>(vals[i]);
    return ret;
  }

  // Intentionally public to allow access to the raw values.
  uint8_t vals[N];
};

template <size_t N> class fp8_e5m2_x {
  static constexpr size_t NExpBits = 5;
  static constexpr size_t NFracBits = 2;
  static constexpr float MaxNormal = 114688.0f;              // 1.75 * 2^16
  static constexpr float MinSubnormal = 0.0000152587890625f; // 2^-16
  static constexpr uint8_t MaxFiniteCode = 0x7C;             // 0.11111.00

  static_assert(N == 1 || N == 2,
                "fp8_e5m2_x: Template argument N must be 1 or 2");

  uint8_t ConvertToFP8(sycl::half h, saturation s) {
#ifdef __SYCL_DEVICE_ONLY__
    return s == saturation::finite
               ? __builtin_spirv_ClampConvertFP16ToE5M2INTEL(h)
               : __builtin_spirv_ConvertFP16ToE5M2EXT(h);
#else
    return detail::ConvertToFP8_CPU<5, 2, sycl::half>(h, rounding::to_even);
#endif
  }

  uint8_t ConvertBF16ToFP8(bfloat16 h, saturation s) {
#ifdef __SYCL_DEVICE_ONLY__
    return s == saturation::finite
               ? __builtin_spirv_ClampConvertBF16ToE5M2INTEL(h)
               : __builtin_spirv_ConvertBF16ToE5M2EXT(h);
#else
    return detail::ConvertToFP8_CPU<5, 2, bfloat16>(h, rounding::to_even);
#endif
  }

  template <typename T> T ConvertFromFP8(uint8_t v) const {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::half hi = __builtin_spirv_ConvertE5M2ToFP16EXT(v);
    return static_cast<T>(hi);
#else
    return detail::ConvertFromFP8_CPU<5, 2, T>(v);
#endif
  }

  bfloat16 ConvertBF16FromFP8(uint8_t v) const {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_spirv_ConvertE5M2ToBF16EXT(v);
#else
    return detail::ConvertFromFP8_CPU<5, 2, bfloat16>(v);
#endif
  }

  void CheckConstraints(rounding r, saturation s) const {
    if (r != rounding::to_even)
      throw std::invalid_argument(
          "fp8_e5m2_x: only rounding::to_even is supported");
  }

public:
  fp8_e5m2_x() = default;
  fp8_e5m2_x(const fp8_e5m2_x &) = default;
  ~fp8_e5m2_x() = default;
  fp8_e5m2_x &operator=(const fp8_e5m2_x &) = default;

  // Construct from pack of half, bfloat16, float, double.
  // Available only when the size of the pack is equal to N.

  // Available only when each type in the pack is half.

  template <typename... Types,
            typename = std::enable_if_t<
                (sizeof...(Types) == N) &&
                (((std::is_same_v<std::decay_t<Types>, half>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, bfloat16>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, float>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, double>) && ...))>>
  explicit fp8_e5m2_x(Types... v) {
    if constexpr (((std::is_same_v<std::decay_t<Types>, bfloat16>) && ...)) {
      const bfloat16 in[N] = {static_cast<bfloat16>(v)...};
      for (size_t i = 0; i < N; ++i)
        vals[i] = ConvertBF16ToFP8(in[i], saturation::finite);
      return;
    }
    const sycl::half in[N] = {v...};
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(in[i], saturation::finite);
  }

  // Construct from an array of half, bfloat16, float, double.

  explicit fp8_e5m2_x(half const (&v)[N], rounding r = rounding::to_even,
                      saturation s = saturation::finite) {
    CheckConstraints(r, s);
    // TODO: optimize with vectorized builtin calls
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i], s);
  }

  explicit fp8_e5m2_x(bfloat16 const (&v)[N], rounding r = rounding::to_even,
                      saturation s = saturation::finite) {
    CheckConstraints(r, s);
    // TODO: optimize with vectorized builtin calls
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertBF16ToFP8(v[i], s);
  }

  explicit fp8_e5m2_x(float const (&v)[N], rounding r = rounding::to_even,
                      saturation s = saturation::finite) {
    CheckConstraints(r, s);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i], s);
  }

  explicit fp8_e5m2_x(double const (&v)[N]) {
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i], saturation::finite);
  }

  // Construct from an marray of half, bfloat16, float, double.

  explicit fp8_e5m2_x(const sycl::marray<sycl::half, N> &v,
                      rounding r = rounding::to_even,
                      saturation s = saturation::finite) {
    CheckConstraints(r, s);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i], s);
  }

  explicit fp8_e5m2_x(const sycl::marray<bfloat16, N> &v,
                      rounding r = rounding::to_even,
                      saturation s = saturation::finite) {
    CheckConstraints(r, s);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertBF16ToFP8(v[i], s);
  }

  explicit fp8_e5m2_x(const sycl::marray<float, N> &v,
                      rounding r = rounding::to_even,
                      saturation s = saturation::finite) {
    CheckConstraints(r, s);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i], s);
  }

  explicit fp8_e5m2_x(const sycl::marray<double, N> &v) {
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i], saturation::finite);
  }

  // Construct with stochastic rounding with user provided seed from an array of
  // half, bfloat16, float.

  explicit fp8_e5m2_x([[maybe_unused]] half const (&in)[N],
                      [[maybe_unused]] const stochastic_seed &seed,
                      [[maybe_unused]] saturation s = saturation::finite) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    for (size_t i = 0; i < N; ++i) {
      if (s == saturation::finite) {
        vals[i] = __builtin_spirv_ClampStochasticRoundFP16ToE5M2INTEL(
            in[i], static_cast<uint32_t>(current_seed), seed.pseed);
      } else {
        vals[i] = __builtin_spirv_StochasticRoundFP16ToE5M2INTEL(
            in[i], static_cast<uint32_t>(current_seed), seed.pseed);
      }
      current_seed = *seed.pseed;
    }
#endif
  }

  explicit fp8_e5m2_x([[maybe_unused]] bfloat16 const (&in)[N],
                      [[maybe_unused]] const stochastic_seed &seed,
                      [[maybe_unused]] saturation s = saturation::finite) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    for (size_t i = 0; i < N; ++i) {
      if (s == saturation::finite) {
        vals[i] = __builtin_spirv_ClampStochasticRoundBF16ToE5M2INTEL(
            in[i], static_cast<uint32_t>(current_seed), seed.pseed);
      } else {
        vals[i] = __builtin_spirv_StochasticRoundBF16ToE5M2INTEL(
            in[i], static_cast<uint32_t>(current_seed), seed.pseed);
      }
      current_seed = *seed.pseed;
    }
#endif
  }

  explicit fp8_e5m2_x([[maybe_unused]] float const (&in)[N],
                      [[maybe_unused]] const stochastic_seed &seed,
                      [[maybe_unused]] saturation s = saturation::finite) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    for (size_t i = 0; i < N; ++i) {
      sycl::half h = static_cast<sycl::half>(in[i]);
      if (s == saturation::finite) {
        vals[i] = __builtin_spirv_ClampStochasticRoundFP16ToE5M2INTEL(
            h, static_cast<uint32_t>(current_seed), seed.pseed);
      } else {
        vals[i] = __builtin_spirv_StochasticRoundFP16ToE5M2INTEL(
            h, static_cast<uint32_t>(current_seed), seed.pseed);
      }
      current_seed = *seed.pseed;
    }
#endif
  }

  // Construct with stochastic rounding with user provided seed from an marray
  // of half, bfloat16, float.

  explicit fp8_e5m2_x([[maybe_unused]] const sycl::marray<half, N> &in,
                      [[maybe_unused]] const stochastic_seed &seed,
                      [[maybe_unused]] saturation s = saturation::finite) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    for (size_t i = 0; i < N; ++i) {
      if (s == saturation::finite) {
        vals[i] = __builtin_spirv_ClampStochasticRoundFP16ToE5M2INTEL(
            in[i], static_cast<uint32_t>(current_seed), seed.pseed);
      } else {
        vals[i] = __builtin_spirv_StochasticRoundFP16ToE5M2INTEL(
            in[i], static_cast<uint32_t>(current_seed), seed.pseed);
      }
      current_seed = *seed.pseed;
    }
#endif
  }

  explicit fp8_e5m2_x([[maybe_unused]] const sycl::marray<bfloat16, N> &in,
                      [[maybe_unused]] const stochastic_seed &seed,
                      [[maybe_unused]] saturation s = saturation::finite) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    for (size_t i = 0; i < N; ++i) {
      if (s == saturation::finite) {
        vals[i] = __builtin_spirv_ClampStochasticRoundBF16ToE5M2INTEL(
            in[i], static_cast<uint32_t>(current_seed), seed.pseed);
      } else {
        vals[i] = __builtin_spirv_StochasticRoundBF16ToE5M2INTEL(
            in[i], static_cast<uint32_t>(current_seed), seed.pseed);
      }
      current_seed = *seed.pseed;
    }
#endif
  }

  explicit fp8_e5m2_x([[maybe_unused]] const sycl::marray<float, N> &in,
                      [[maybe_unused]] const stochastic_seed &seed,
                      [[maybe_unused]] saturation s = saturation::finite) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    for (size_t i = 0; i < N; ++i) {
      sycl::half h = static_cast<sycl::half>(in[i]);
      if (s == saturation::finite) {
        vals[i] = __builtin_spirv_ClampStochasticRoundFP16ToE5M2INTEL(
            h, static_cast<uint32_t>(current_seed), seed.pseed);
      } else {
        vals[i] = __builtin_spirv_StochasticRoundFP16ToE5M2INTEL(
            h, static_cast<uint32_t>(current_seed), seed.pseed);
      }
      current_seed = *seed.pseed;
    }
#endif
  }

  // Construct from integer types.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e5m2_x(short val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e5m2_x(int val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e5m2_x(long val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e5m2_x(long long val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e5m2_x(unsigned short val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e5m2_x(unsigned int val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e5m2_x(unsigned long val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e5m2_x(unsigned long long val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  // Assign (operator) from half, bfloat16, float, double, and integer types.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(sycl::half val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(bfloat16 val) {
    vals[0] = ConvertBF16ToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(float val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(double val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(short val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(int val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(long val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(long long val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(unsigned short val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(unsigned int val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(unsigned long val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e5m2_x &operator=(unsigned long long val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  // Convert to half, bfloat16, float, double.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator half() const {
    return ConvertFromFP8<sycl::half>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bfloat16() const {
    return ConvertBF16FromFP8(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator float() const {
    return ConvertFromFP8<float>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator double() const {
    return ConvertFromFP8<double>(vals[0]);
  }

  // Convert to integer types.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator char() const {
    return ConvertFromFP8<char>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator signed char() const {
    return ConvertFromFP8<signed char>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator short() const {
    return ConvertFromFP8<short>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator int() const {
    return ConvertFromFP8<int>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator long() const {
    return ConvertFromFP8<long>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator long long() const {
    return ConvertFromFP8<long long>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned char() const {
    return ConvertFromFP8<unsigned char>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned short() const {
    return ConvertFromFP8<unsigned short>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned int() const {
    return ConvertFromFP8<unsigned int>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned long() const {
    return ConvertFromFP8<unsigned long>(vals[0]);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned long long() const {
    return ConvertFromFP8<unsigned long long>(vals[0]);
  }

  // Convert to bool
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bool() const {
    // false iff +0 or -0; otherwise true.
    return vals[0] != 0x00 && vals[0] != 0x80;
  }

  explicit operator sycl::marray<sycl::half, N>() const {
    sycl::marray<sycl::half, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] = ConvertFromFP8<sycl::half>(vals[i]);
    return out;
  }
  explicit operator sycl::marray<bfloat16, N>() const {
    sycl::marray<bfloat16, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] = ConvertBF16FromFP8(vals[i]);
    return out;
  }
  explicit operator sycl::marray<float, N>() const {
    sycl::marray<float, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] = ConvertFromFP8<float>(vals[i]);
    return out;
  }

  // Intentionally public to allow access to the raw values.

  uint8_t vals[N];
};

template <size_t N> class fp8_e8m0_x {
  static_assert(N == 1 || N == 2,
                "fp8_e8m0_x: Template argument N must be 1 or 2");

  void CheckConstraints(rounding r) const {

    if (r != rounding::upward && r != rounding::toward_zero)
      throw std::invalid_argument("fp8_e8m0_x: only rounding::upward and "
                                  "rounding::toward_zero are supported");
  }

public:
  fp8_e8m0_x() = default;
  fp8_e8m0_x(const fp8_e8m0_x &) = default;
  ~fp8_e8m0_x() = default;
  fp8_e8m0_x &operator=(const fp8_e8m0_x &) = default;

  template <typename... Types,
            typename = std::enable_if_t<
                (sizeof...(Types) == N) &&
                (((std::is_same_v<std::decay_t<Types>, half>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, bfloat16>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, float>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, double>) && ...))>>
  explicit fp8_e8m0_x(Types... v) {
    using InT = std::common_type_t<std::decay_t<Types>...>;
    const InT in[N] = {v...};
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], rounding::upward,
                                          saturation::finite);
  }

  explicit fp8_e8m0_x(half const (&in)[N], rounding r = rounding::upward) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], r, saturation::finite);
  }

  explicit fp8_e8m0_x(bfloat16 const (&in)[N], rounding r = rounding::upward) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], r, saturation::finite);
  }

  explicit fp8_e8m0_x(float const (&in)[N], rounding r = rounding::upward) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], r, saturation::finite);
  }

  explicit fp8_e8m0_x(double const (&in)[N]) {
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], rounding::upward,
                                          saturation::finite);
  }

  explicit fp8_e8m0_x(const marray<half, N> &in,
                      rounding r = rounding::upward) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], r, saturation::finite);
  }

  explicit fp8_e8m0_x(const marray<bfloat16, N> &in,
                      rounding r = rounding::upward) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], r, saturation::finite);
  }

  explicit fp8_e8m0_x(const marray<float, N> &in,
                      rounding r = rounding::upward) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], r, saturation::finite);
  }

  explicit fp8_e8m0_x(const marray<double, N> &in) {
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertToE8M0_CPU(in[i], rounding::upward,
                                          saturation::finite);
  }

  // Construct from integer types.
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e8m0_x(short val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e8m0_x(int val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e8m0_x(long val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e8m0_x(long long val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e8m0_x(unsigned short val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e8m0_x(unsigned int val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e8m0_x(unsigned long val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp8_e8m0_x(unsigned long long val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(half val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(bfloat16 val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(float val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(double val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(short val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(int val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(long val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(long long val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(unsigned short val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(unsigned int val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(unsigned long val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp8_e8m0_x &operator=(unsigned long long val) {
    vals[0] =
        detail::ConvertToE8M0_CPU(val, rounding::upward, saturation::finite);
    return *this;
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator half() const {
    return detail::ConvertFromE8M0_CPU<half>(vals[0], rounding::to_even);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bfloat16() const {
    return detail::ConvertFromE8M0_CPU<bfloat16>(vals[0], rounding::to_even);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator float() const {
    return detail::ConvertFromE8M0_CPU<float>(vals[0], rounding::to_even);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator double() const {
    return detail::ConvertFromE8M0_CPU<double>(vals[0], rounding::to_even);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator char() const {
    return detail::ConvertFromE8M0_CPU<char>(vals[0], rounding::toward_zero);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator signed char() const {
    return detail::ConvertFromE8M0_CPU<signed char>(vals[0],
                                                    rounding::toward_zero);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator short() const {
    return detail::ConvertFromE8M0_CPU<short>(vals[0], rounding::toward_zero);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator int() const {
    return detail::ConvertFromE8M0_CPU<int>(vals[0], rounding::toward_zero);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator long() const {
    return detail::ConvertFromE8M0_CPU<long>(vals[0], rounding::toward_zero);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator long long() const {
    return detail::ConvertFromE8M0_CPU<long long>(vals[0],
                                                  rounding::toward_zero);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned char() const {
    return detail::ConvertFromE8M0_CPU<unsigned char>(vals[0],
                                                      rounding::toward_zero);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned short() const {
    return detail::ConvertFromE8M0_CPU<unsigned short>(vals[0],
                                                       rounding::toward_zero);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned int() const {
    return detail::ConvertFromE8M0_CPU<unsigned int>(vals[0],
                                                     rounding::toward_zero);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned long() const {
    return detail::ConvertFromE8M0_CPU<unsigned long>(vals[0],
                                                      rounding::toward_zero);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned long long() const {
    return detail::ConvertFromE8M0_CPU<unsigned long long>(
        vals[0], rounding::toward_zero);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bool() const {
    return true;
  }

  explicit operator sycl::marray<half, N>() const {
    sycl::marray<half, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] = detail::ConvertFromE8M0_CPU<half>(vals[i], rounding::to_even);
    return out;
  }
  explicit operator sycl::marray<bfloat16, N>() const {
    sycl::marray<bfloat16, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] =
          detail::ConvertFromE8M0_CPU<bfloat16>(vals[i], rounding::to_even);
    return out;
  }
  explicit operator sycl::marray<float, N>() const {
    sycl::marray<float, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] = detail::ConvertFromE8M0_CPU<float>(vals[i], rounding::to_even);
    return out;
  }

  // Intentionally public to allow access to the raw values.

  uint8_t vals[N];
};

using fp8_e4m3 = fp8_e4m3_x<1>;
using fp8_e4m3_x2 = fp8_e4m3_x<2>;
using fp8_e5m2 = fp8_e5m2_x<1>;
using fp8_e5m2_x2 = fp8_e5m2_x<2>;
using fp8_e8m0 = fp8_e8m0_x<1>;
using fp8_e8m0_x2 = fp8_e8m0_x<2>;

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
