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
#include <cstring>
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

template <typename T> static inline int BitWidth(T x) noexcept {
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
  static constexpr uint16_t InfBits = 0x7C00u;
  static constexpr uint16_t QuietNaNBits = 0x7E00u;
};

template <> struct DirectBinary16Traits<sycl::ext::oneapi::bfloat16> {
  static constexpr uint16_t SignMask = 0x8000u;
  static constexpr uint16_t InfBits = 0x7F80u;
  static constexpr uint16_t QuietNaNBits = 0x7FC0u;
};

template <typename ToT> static constexpr inline ToT MakeDirectNaN() noexcept {
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
static constexpr inline ToT MakeDirectInf(bool negative) noexcept {
  if constexpr (std::is_same_v<ToT, sycl::half> ||
                std::is_same_v<ToT, sycl::ext::oneapi::bfloat16>) {
    using Traits = DirectBinary16Traits<ToT>;
    const uint16_t sign = negative ? Traits::SignMask : 0u;
    return sycl::bit_cast<ToT>(static_cast<uint16_t>(sign | Traits::InfBits));
  } else if constexpr (std::numeric_limits<ToT>::has_infinity) {
    return negative ? -std::numeric_limits<ToT>::infinity()
                    : std::numeric_limits<ToT>::infinity();
  } else {
    return ToT{};
  }
}

template <typename T> struct SourceTraits;

template <> struct SourceTraits<float> {
  using UInt = uint32_t;
  static constexpr size_t ExpBits = 8;
  static constexpr size_t FracBits = 23;
  static constexpr int Bias = 127;
};

template <> struct SourceTraits<sycl::half> {
  using UInt = uint16_t;
  static constexpr size_t ExpBits = 5;
  static constexpr size_t FracBits = 10;
  static constexpr int Bias = 15;
};

template <> struct SourceTraits<sycl::ext::oneapi::bfloat16> {
  using UInt = uint16_t;
  static constexpr size_t ExpBits = 8;
  static constexpr size_t FracBits = 7;
  static constexpr int Bias = 127;
};

template <> struct SourceTraits<double> {
  using UInt = uint64_t;
  static constexpr size_t ExpBits = 11;
  static constexpr size_t FracBits = 52;
  static constexpr int Bias = 1023;
};

template <> struct SourceTraits<char> {
  using UInt = uint8_t;
  using UnsignedT = std::make_unsigned_t<char>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = std::numeric_limits<char>::is_signed;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<signed char> {
  using UInt = uint8_t;
  using UnsignedT = std::make_unsigned_t<signed char>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = true;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<unsigned char> {
  using UInt = uint8_t;
  using UnsignedT = std::make_unsigned_t<unsigned char>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = false;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<short> {
  using UInt = uint16_t;
  using UnsignedT = std::make_unsigned_t<short>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = true;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<int> {
  using UInt = uint32_t;
  using UnsignedT = std::make_unsigned_t<int>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = true;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<long> {
  using UInt = std::make_unsigned_t<long>;
  using UnsignedT = std::make_unsigned_t<long>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = true;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<long long> {
  using UInt = uint64_t;
  using UnsignedT = std::make_unsigned_t<long long>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = true;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<unsigned short> {
  using UInt = uint16_t;
  using UnsignedT = std::make_unsigned_t<unsigned short>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = false;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<unsigned int> {
  using UInt = uint32_t;
  using UnsignedT = std::make_unsigned_t<unsigned int>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = false;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<unsigned long> {
  using UInt = std::make_unsigned_t<unsigned long>;
  using UnsignedT = std::make_unsigned_t<unsigned long>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = false;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <> struct SourceTraits<unsigned long long> {
  using UInt = uint64_t;
  using UnsignedT = std::make_unsigned_t<unsigned long long>;

  static constexpr bool IsIntegral = true;
  static constexpr bool IsSigned = false;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

template <int Ebits, int Mbits> struct FP8FiniteFormatTraits {
  static_assert((Ebits == 4 && Mbits == 3) || (Ebits == 5 && Mbits == 2),
                "Unsupported FP8 finite format");

  static constexpr uint8_t ExpAllOnes =
      static_cast<uint8_t>((1u << Ebits) - 1u);
  static constexpr uint8_t MaxFrac = static_cast<uint8_t>((1u << Mbits) - 1u);
  static constexpr int Bias = (1 << (Ebits - 1)) - 1;
  static constexpr int Emin = 1 - Bias;
  static constexpr bool HasInfinity = (Ebits == 5 && Mbits == 2);
  static constexpr uint8_t MaxFiniteExpField =
      HasInfinity ? static_cast<uint8_t>(ExpAllOnes - 1u) : ExpAllOnes;
  static constexpr uint8_t MaxFiniteFracField =
      (Ebits == 4 && Mbits == 3) ? static_cast<uint8_t>(MaxFrac - 1u) : MaxFrac;
  static constexpr uint8_t MaxFiniteCode =
      static_cast<uint8_t>((MaxFiniteExpField << Mbits) | MaxFiniteFracField);
  static constexpr uint8_t NaNCode =
      static_cast<uint8_t>((ExpAllOnes << Mbits) | MaxFrac);
  static constexpr uint8_t InfinityCode =
      static_cast<uint8_t>(ExpAllOnes << Mbits);
  static constexpr int MaxFiniteExp =
      static_cast<int>(MaxFiniteExpField) - Bias;
  static constexpr uint64_t MinNormalMantissa = uint64_t{1} << Mbits;
  static constexpr uint64_t OverflowMantissa = uint64_t{1} << (Mbits + 1);
  static constexpr uint64_t MaxFiniteMantissa =
      MinNormalMantissa + MaxFiniteFracField;
};

template <typename T, typename Traits = SourceTraits<T>>
static inline uint8_t ConvertIntToE8M0_CPU(T f, rounding R,
                                           saturation S) noexcept {
  using UnsignedT = typename Traits::UnsignedT;
  UnsignedT magnitude = f < 0 ? -f : f;

  if (magnitude == 0)
    return 0x00u;

  int lowerExp = BitWidth(static_cast<uint64_t>(magnitude)) - 1;
  bool isExactPowerOfTwo = (magnitude & (magnitude - 1)) == 0;

  bool roundUp = false;
  switch (R) {
  case rounding::toward_zero:
    break;
  case rounding::upward:
    roundUp = !isExactPowerOfTwo;
    break;
  case rounding::to_even: {
    if (!isExactPowerOfTwo) {
      const uint64_t leading = uint64_t{1} << lowerExp;
      const uint64_t twice = 2ull * static_cast<uint64_t>(magnitude);
      const uint64_t midpoint = 3ull * leading;
      if (twice > midpoint || (twice == midpoint && (lowerExp & 1) != 0))
        roundUp = true;
    }
    break;
  }
  }
  return static_cast<uint8_t>(127 + lowerExp + (roundUp ? 1 : 0));
}

template <int Ebits = 4, int Mbits = 3, typename T,
          typename Traits = SourceTraits<T>>
static inline uint8_t ConvertIntToFP8_CPU(T f, rounding R,
                                          saturation S) noexcept {
  using UnsignedT = typename Traits::UnsignedT;
  using Format = FP8FiniteFormatTraits<Ebits, Mbits>;

  auto getOverflowCode = [&]() -> uint8_t {
    if (S == saturation::finite)
      return Format::MaxFiniteCode;
    if constexpr (Format::HasInfinity)
      return Format::InfinityCode;
    return Format::NaNCode;
  };

  const uint8_t sign =
      (Traits::IsSigned && f < 0) ? static_cast<uint8_t>(0x80u) : 0u;
  UnsignedT magnitude = 0;

  if constexpr (Traits::IsSigned) {
    const UnsignedT bits = static_cast<UnsignedT>(f);
    magnitude = f < 0 ? static_cast<UnsignedT>(UnsignedT{0} - bits) : bits;
  } else {
    magnitude = static_cast<UnsignedT>(f);
  }

  if (magnitude == 0)
    return sign;

  int unbiasedExp = BitWidth(static_cast<uint64_t>(magnitude)) - 1;
  if (unbiasedExp > Format::MaxFiniteExp)
    return static_cast<uint8_t>(sign | getOverflowCode());

  const int shift = unbiasedExp - Mbits;
  uint64_t mantissa = 0u;
  if (shift <= 0) {
    mantissa = static_cast<uint64_t>(magnitude) << (-shift);
  } else {
    const uint64_t truncated = static_cast<uint64_t>(magnitude) >> shift;
    const uint64_t remainderMask = (uint64_t{1} << shift) - 1u;
    const uint64_t remainder = static_cast<uint64_t>(magnitude) & remainderMask;

    mantissa = truncated;
    if (remainder != 0u) {
      if (R == rounding::upward) {
        if (sign == 0u)
          ++mantissa;
      } else if (R == rounding::to_even) {
        const uint64_t half = uint64_t{1} << (shift - 1);
        if (remainder > half ||
            (remainder == half && (truncated & uint64_t{1}) != 0u)) {
          ++mantissa;
        }
      }
    }
  }

  if (mantissa >= Format::OverflowMantissa) {
    mantissa = Format::MinNormalMantissa;
    ++unbiasedExp;
  }

  if (unbiasedExp > Format::MaxFiniteExp)
    return static_cast<uint8_t>(sign | getOverflowCode());

  if (unbiasedExp == Format::MaxFiniteExp &&
      mantissa > Format::MaxFiniteMantissa)
    return static_cast<uint8_t>(sign | getOverflowCode());

  const uint8_t expField = static_cast<uint8_t>(unbiasedExp + Format::Bias);
  const uint8_t fracField =
      static_cast<uint8_t>(mantissa - Format::MinNormalMantissa);
  return static_cast<uint8_t>(sign | static_cast<uint8_t>(expField << Mbits) |
                              fracField);
}

template <int Ebits = 4, int Mbits = 3, typename T,
          typename Traits = SourceTraits<T>>
static inline uint8_t ConvertFloatToFP8_CPU(T f, rounding R,
                                            saturation S) noexcept {
  using UInt = typename Traits::UInt;
  using Format = FP8FiniteFormatTraits<Ebits, Mbits>;

  constexpr UInt SignMask = UInt{1} << (Traits::ExpBits + Traits::FracBits);
  constexpr UInt FracMask = (UInt{1} << Traits::FracBits) - UInt{1};
  constexpr UInt ExpMask = ((UInt{1} << Traits::ExpBits) - UInt{1})
                           << Traits::FracBits;
  constexpr UInt ExpAllOnes = (UInt{1} << Traits::ExpBits) - UInt{1};

  auto getOverflowCode = [&](uint8_t sign) -> uint8_t {
    if (S == saturation::finite)
      return static_cast<uint8_t>(sign | Format::MaxFiniteCode);
    if constexpr (Format::HasInfinity)
      return static_cast<uint8_t>(sign | Format::InfinityCode);
    return static_cast<uint8_t>(sign | Format::NaNCode);
  };

  UInt bits;
  __builtin_memcpy(&bits, &f, sizeof(bits));

  const uint8_t sign = (bits & SignMask) ? 0x80u : 0x00u;
  bits &= ~SignMask;

  const UInt exp = (bits & ExpMask) >> Traits::FracBits;
  const UInt frac = bits & FracMask;

  if (exp == ExpAllOnes) {
    if (frac != 0u)
      return static_cast<uint8_t>(sign | Format::NaNCode);
    return getOverflowCode(sign);
  }

  if (exp == 0u && frac == 0u)
    return sign;

  uint64_t significand = 0u;
  int leadingBit = 0;
  int unbiasedExp = 0;

  if (exp != 0u) {
    significand =
        (uint64_t{1} << Traits::FracBits) | static_cast<uint64_t>(frac);
    leadingBit = static_cast<int>(Traits::FracBits);
    unbiasedExp = static_cast<int>(exp) - Traits::Bias;
  } else {
    significand = static_cast<uint64_t>(frac);
    uint64_t tmp = significand;
    leadingBit = -1;
    while (tmp != 0u) {
      ++leadingBit;
      tmp >>= 1;
    }
    unbiasedExp =
        1 - Traits::Bias - static_cast<int>(Traits::FracBits) + leadingBit;
  }

  auto roundShiftRight = [&](uint64_t value, int shift) -> uint64_t {
    if (shift <= 0)
      return value;

    if (shift >= 64) {
      if (R == rounding::upward && sign == 0u && value != 0u)
        return 1u;
      return 0u;
    }

    const uint64_t truncated = value >> shift;
    const uint64_t remainderMask = (uint64_t{1} << shift) - 1u;
    const uint64_t remainder = value & remainderMask;

    if (remainder == 0u)
      return truncated;

    if (R == rounding::toward_zero)
      return truncated;

    if (R == rounding::upward)
      return sign == 0u ? truncated + 1u : truncated;

    const uint64_t half = uint64_t{1} << (shift - 1);
    if (remainder > half)
      return truncated + 1u;
    if (remainder < half)
      return truncated;
    return (truncated & 1u) != 0u ? truncated + 1u : truncated;
  };

  if (unbiasedExp > Format::MaxFiniteExp)
    return getOverflowCode(sign);

  if (unbiasedExp == Format::MaxFiniteExp) {
    const uint64_t lhs = significand << Mbits;
    const uint64_t rhs = Format::MaxFiniteMantissa << leadingBit;
    if (lhs > rhs)
      return getOverflowCode(sign);
  }

  if (unbiasedExp < Format::Emin) {
    const int shift = leadingBit - unbiasedExp - Format::Bias - Mbits + 1;
    uint64_t mantissa = shift > 0 ? roundShiftRight(significand, shift)
                                  : (significand << (-shift));

    if (mantissa == 0u)
      return sign;

    if (mantissa >= Format::MinNormalMantissa)
      return static_cast<uint8_t>(sign | (uint8_t{1} << Mbits));

    return static_cast<uint8_t>(sign | static_cast<uint8_t>(mantissa));
  }

  const int shift = leadingBit - Mbits;
  uint64_t mantissa = shift > 0 ? roundShiftRight(significand, shift)
                                : (significand << (-shift));

  if (mantissa >= Format::OverflowMantissa) {
    mantissa = Format::MinNormalMantissa;
    ++unbiasedExp;
  }

  if (unbiasedExp > Format::MaxFiniteExp)
    return getOverflowCode(sign);

  if (unbiasedExp == Format::MaxFiniteExp &&
      mantissa > Format::MaxFiniteMantissa)
    return getOverflowCode(sign);

  const uint8_t expField = static_cast<uint8_t>(unbiasedExp + Format::Bias);
  const uint8_t fracField =
      static_cast<uint8_t>(mantissa - Format::MinNormalMantissa);
  return static_cast<uint8_t>(sign | static_cast<uint8_t>(expField << Mbits) |
                              fracField);
}

template <typename T, typename Traits = SourceTraits<T>>
static inline uint8_t ConvertFloatToE8M0_CPU(T f, rounding R,
                                             saturation S) noexcept {
  using UInt = typename Traits::UInt;
  constexpr UInt SignMask = UInt{1} << (Traits::ExpBits + Traits::FracBits);
  constexpr UInt FracMask = (UInt{1} << Traits::FracBits) - 1;
  constexpr UInt ExpMask = ((UInt{1} << Traits::ExpBits) - 1)
                           << Traits::FracBits;
  constexpr UInt ExpAllOnes = (UInt{1} << Traits::ExpBits) - 1;
  constexpr uint8_t NaNCode = 0xFF;
  constexpr uint8_t MaxFiniteCode = 0xFE;
  constexpr int TargetBias = 127;
  constexpr int TargetEmin = -127;
  constexpr int TargetEmax = 127;

  UInt h;
  __builtin_memcpy(&h, &f, sizeof(h));
  h &= ~SignMask;

  UInt exp = (h & ExpMask) >> Traits::FracBits;
  UInt frac = h & FracMask;

  if (exp == ExpAllOnes) {
    if (frac != 0u)
      return NaNCode;
    return (S == saturation::finite) ? MaxFiniteCode : NaNCode;
  }

  if (exp == 0u && frac == 0u)
    return 0x00u;

  uint64_t significand = 0u;
  int leadingBit = 0;
  int lowerExp = 0;
  bool isExactPowerOfTwo = false;

  if (exp != 0u) {
    significand =
        (uint64_t{1} << Traits::FracBits) | static_cast<uint64_t>(frac);
    leadingBit = static_cast<int>(Traits::FracBits);
    lowerExp = static_cast<int>(exp) - Traits::Bias;
    isExactPowerOfTwo = frac == 0u;
  } else {
    significand = static_cast<uint64_t>(frac);
    leadingBit = BitWidth(significand) - 1;
    lowerExp =
        1 - Traits::Bias - static_cast<int>(Traits::FracBits) + leadingBit;
    isExactPowerOfTwo = (significand & (significand - 1u)) == 0u;
  }

  if (lowerExp < TargetEmin)
    return 0x00u;

  bool roundUp = false;

  switch (R) {
  case rounding::toward_zero:
    break;
  case rounding::upward:
    roundUp = !isExactPowerOfTwo;
    break;
  case rounding::to_even: {
    if (!isExactPowerOfTwo) {
      const uint64_t twiceSignificand = 2ull * significand;
      const uint64_t midpoint = 3ull * (uint64_t{1} << leadingBit);
      if (twiceSignificand > midpoint) {
        roundUp = true;
      } else if (twiceSignificand == midpoint && (lowerExp & 1) != 0) {
        roundUp = true;
      }
    }
    break;
  }
  }

  int encodedExp = lowerExp + (roundUp ? 1 : 0);
  if (encodedExp > TargetEmax)
    return (S == saturation::finite) ? MaxFiniteCode : NaNCode;

  return static_cast<uint8_t>(encodedExp + TargetBias);
}

template <typename Traits, typename = void>
struct HasE8M0FloatTraits : std::false_type {};

template <typename Traits>
struct HasE8M0FloatTraits<
    Traits, std::void_t<decltype(Traits::ExpBits), decltype(Traits::FracBits),
                        decltype(Traits::Bias)>> : std::true_type {};

template <typename Traits, typename = void>
struct HasE8M0IntegralTraits : std::false_type {};

template <typename Traits>
struct HasE8M0IntegralTraits<
    Traits,
    std::void_t<typename Traits::UnsignedT, decltype(Traits::IsIntegral),
                decltype(Traits::IsSigned), decltype(Traits::ValueBits)>>
    : std::true_type {};

template <int Ebits, int Mbits, typename ToT,
          typename Traits = SourceTraits<ToT>>
static inline ToT ConvertFromFP8ToBinaryFloat_CPU(uint8_t code,
                                                  rounding R) noexcept {
  static_assert((Ebits == 8 && Mbits == 0) || (Ebits == 4 && Mbits == 3) ||
                    (Ebits == 5 && Mbits == 2),
                "Unsupported FP8 decode combination");

  constexpr int SrcBias = (1 << (Ebits - 1)) - 1;
  constexpr int SrcEmin = 1 - SrcBias;
  constexpr uint8_t SrcExpAllOnes = static_cast<uint8_t>((1u << Ebits) - 1u);
  constexpr uint8_t SrcFracMask =
      (Mbits == 0) ? 0u : static_cast<uint8_t>((1u << Mbits) - 1u);

  bool negative = false;
  uint32_t significand = 0u;
  int exp2 = 0;
  bool isNaN = false;
  bool isInf = false;

  if constexpr (Ebits == 8 && Mbits == 0) {
    if (code == 0xFFu) {
      isNaN = true;
    } else {
      significand = 1u;
      exp2 = static_cast<int>(code) - SrcBias;
    }
  } else {
    negative = (code & 0x80u) != 0u;
    const uint8_t exp = static_cast<uint8_t>((code >> Mbits) & SrcExpAllOnes);
    const uint8_t frac = static_cast<uint8_t>(code & SrcFracMask);

    if (exp == SrcExpAllOnes) {
      if constexpr (Ebits == 5 && Mbits == 2) {
        if (frac == 0u)
          isInf = true;
        else
          isNaN = true;
      } else if (frac == SrcFracMask) {
        isNaN = true;
      } else {
        significand = static_cast<uint32_t>((1u << Mbits) + frac);
        exp2 = static_cast<int>(exp) - SrcBias;
      }
    } else if (exp == 0u) {
      if (frac == 0u)
        significand = 0u;
      else {
        significand = frac;
        exp2 = SrcEmin;
      }
    } else {
      significand = static_cast<uint32_t>((1u << Mbits) + frac);
      exp2 = static_cast<int>(exp) - SrcBias;
    }
  }

  if constexpr (HasE8M0FloatTraits<Traits>::value) {
    using UInt = typename Traits::UInt;

    constexpr UInt ExpAllOnes = ((UInt{1} << Traits::ExpBits) - UInt{1})
                                << Traits::FracBits;
    constexpr UInt FracMask = (UInt{1} << Traits::FracBits) - UInt{1};
    constexpr UInt QuietNaNBit = UInt{1} << (Traits::FracBits - 1);
    constexpr UInt MaxFiniteBits =
        (ExpAllOnes - (UInt{1} << Traits::FracBits)) | FracMask;
    constexpr int MinNormalExp = 1 - Traits::Bias;
    constexpr int MinSubnormalExp =
        MinNormalExp - static_cast<int>(Traits::FracBits);
    constexpr int MaxNormalExp =
        static_cast<int>((UInt{1} << Traits::ExpBits) - UInt{2}) - Traits::Bias;

    UInt bits = 0;
    if (isNaN) {
      bits = ExpAllOnes | QuietNaNBit;
    } else if (isInf) {
      bits =
          (negative ? (UInt{1} << (Traits::ExpBits + Traits::FracBits)) : 0u) |
          ExpAllOnes;
    } else if (significand == 0u) {
      bits = negative ? (UInt{1} << (Traits::ExpBits + Traits::FracBits)) : 0u;
    } else {
      const int sigBits = BitWidth(significand);
      const int unbiasedExp = exp2 + sigBits - 1 - Mbits;
      const UInt signBit =
          negative ? (UInt{1} << (Traits::ExpBits + Traits::FracBits)) : 0u;

      if (unbiasedExp > MaxNormalExp) {
        bits = signBit |
               ((R == rounding::toward_zero) ? MaxFiniteBits : ExpAllOnes);
      } else if (unbiasedExp >= MinNormalExp) {
        const int shift = static_cast<int>(Traits::FracBits) - (sigBits - 1);
        const UInt aligned = static_cast<UInt>(significand) << shift;
        const UInt expField = static_cast<UInt>(unbiasedExp + Traits::Bias)
                              << Traits::FracBits;
        bits = signBit | expField | (aligned & FracMask);
      } else if (unbiasedExp >= MinSubnormalExp) {
        const int subShift =
            exp2 - Mbits - MinNormalExp + static_cast<int>(Traits::FracBits);
        const UInt fracField = static_cast<UInt>(significand) << subShift;
        bits = signBit | fracField;
      } else if (R == rounding::upward && !negative) {
        bits = UInt{1};
      }
    }

    return __builtin_bit_cast(ToT, bits);
  } else if constexpr (HasE8M0IntegralTraits<Traits>::value &&
                       Traits::IsIntegral) {
    using UnsignedT = typename Traits::UnsignedT;

    if (isNaN || isInf)
      return ToT{};

    if (significand == 0u)
      return ToT{};

    const int shift = exp2 - Mbits;
    uint64_t magnitude = 0u;

    if (shift >= 0) {
      if (shift >= 64)
        return ToT{};
      magnitude = static_cast<uint64_t>(significand) << shift;
    } else {
      const int rshift = -shift;
      if (rshift >= 64) {
        if (R == rounding::upward && !negative)
          magnitude = 1u;
      } else {
        magnitude = static_cast<uint64_t>(significand) >> rshift;
        const uint64_t remainderMask = (uint64_t{1} << rshift) - 1u;
        const uint64_t remainder =
            static_cast<uint64_t>(significand) & remainderMask;

        if (remainder != 0u) {
          if (R == rounding::upward) {
            if (!negative)
              ++magnitude;
          } else if (R == rounding::to_even) {
            const uint64_t half = uint64_t{1} << (rshift - 1);
            if (remainder > half ||
                (remainder == half && (magnitude & 1u) != 0u)) {
              ++magnitude;
            }
          }
        }
      }
    }

    if (magnitude == 0u)
      return ToT{};

    if (BitWidth(magnitude) > Traits::ValueBits)
      return ToT{};

    const UnsignedT narrowed = static_cast<UnsignedT>(magnitude);
    if constexpr (Traits::IsSigned)
      return static_cast<ToT>(negative ? -static_cast<ToT>(narrowed)
                                       : static_cast<ToT>(narrowed));
    return static_cast<ToT>(narrowed);
  }

  if (isNaN)
    return MakeDirectNaN<ToT>();
  if (isInf)
    return MakeDirectInf<ToT>(negative);

  return ToT{};
}

template <typename ToT>
static inline ToT ConvertFromE8M0_CPU(uint8_t code, rounding R) noexcept {
  using Traits = SourceTraits<ToT>;

  if constexpr (HasE8M0FloatTraits<Traits>::value ||
                HasE8M0IntegralTraits<Traits>::value) {
    return ConvertFromFP8ToBinaryFloat_CPU<8, 0, ToT>(code, R);
  }

  return ToT{};
}

template <typename T, typename... Ts>
struct IsOneOf : std::disjunction<std::is_same<T, Ts>...> {};

template <typename T>
struct IsSyclFpType : IsOneOf<std::decay_t<T>, sycl::half,
                              sycl::ext::oneapi::bfloat16, float, double> {};

template <typename T>
inline constexpr bool IsSyclFpTypeV = IsSyclFpType<T>::value;

template <size_t N, typename... Types>
struct SyclfpVariadic
    : std::bool_constant<
          (sizeof...(Types) == N) &&
          (((std::is_same_v<std::decay_t<Types>, half>) && ...) ||
           ((std::is_same_v<std::decay_t<Types>, bfloat16>) && ...) ||
           ((std::is_same_v<std::decay_t<Types>, float>) && ...) ||
           ((std::is_same_v<std::decay_t<Types>, double>) && ...))> {};

template <size_t N, typename... Types>
inline constexpr bool SyclfpVariadicV = SyclfpVariadic<N, Types...>::value;

} // namespace detail

template <size_t N> class fp8_e4m3_x {
  static constexpr size_t NExpBits = 4;
  static constexpr size_t NFracBits = 3;

  static_assert(N == 1 || N == 2,
                "fp8_e4m3_x: Template argument N must be 1 or 2");

  template <typename T> uint8_t ConvertToFP8(T h) {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<std::decay_t<T>, bfloat16>)
      return __builtin_spirv_ClampConvertBF16ToE4M3INTEL(h);
    else
      return __builtin_spirv_ClampConvertFP16ToE4M3INTEL(h);
#else
    if constexpr (detail::IsSyclFpTypeV<T>) {
      return detail::ConvertFloatToFP8_CPU<NExpBits, NFracBits, T>(
          h, rounding::to_even, saturation::finite);
    } else if constexpr (std::is_integral_v<std::decay_t<T>>) {
      return detail::ConvertIntToFP8_CPU<NExpBits, NFracBits, T>(
          h, rounding::to_even, saturation::finite);
    }
#endif
  }

  template <typename T>
  T ConvertFromFP8(uint8_t v, rounding r = rounding::to_even) const {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<std::decay_t<T>, bfloat16>)
      return __builtin_spirv_ConvertE4M3ToBF16EXT(v);
    else {
      sycl::half hi = __builtin_spirv_ConvertE4M3ToFP16EXT(v);
      return static_cast<T>(hi);
    }
#else
    return detail::ConvertFromFP8ToBinaryFloat_CPU<NExpBits, NFracBits, T>(v,
                                                                           r);
#endif
  }

  void CheckConstraints(rounding r) const {
    assert(r == rounding::to_even &&
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
            typename = std::enable_if_t<detail::SyclfpVariadicV<N, Types...>>>
  explicit fp8_e4m3_x(Types... v) {
    using InT = std::common_type_t<std::decay_t<Types>...>;
    const InT in[N] = {v...};
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(in[i]);
  }

  // Construct from an array of half, bfloat16, float, double.
  template <typename T, typename = std::enable_if_t<detail::IsSyclFpTypeV<T>>>
  explicit fp8_e4m3_x(T const (&v)[N], rounding r = rounding::to_even) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i]);
  }

  // Construct from an marray of half, bfloat16, float, double.
  template <typename T, typename = std::enable_if_t<detail::IsSyclFpTypeV<T>>>
  explicit fp8_e4m3_x(const sycl::marray<T, N> &v,
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i]);
  }

  // Construct from integer types.
  // Available only when N==1.

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && std::is_integral_v<T>>>
  explicit fp8_e4m3_x(T val) {
    vals[0] = ConvertToFP8(val);
  }
  // Assign (operator) from half, bfloat16, float, double, and integer types.
  // Available only when N==1.

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && (detail::IsSyclFpTypeV<T> ||
                                                   std::is_integral_v<T>)>>
  fp8_e4m3_x &operator=(T val) {
    vals[0] = ConvertToFP8(val);
    return *this;
  }
  // Convert to half, bfloat16, float, double and integer types
  // Available only when N==1.

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && (detail::IsSyclFpTypeV<T> ||
                                                   std::is_integral_v<T>)>>
  explicit operator T() const {
    if constexpr (std::is_integral_v<T>)
      return ConvertFromFP8<T>(vals[0], rounding::toward_zero);
    else
      return ConvertFromFP8<T>(vals[0]);
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
  template <typename T, typename = std::enable_if_t<detail::IsOneOf<
                            std::decay_t<T>, sycl::half,
                            sycl::ext::oneapi::bfloat16, float>::value>>
  explicit operator sycl::marray<T, N>() const {
    sycl::marray<T, N> ret;
    for (size_t i = 0; i < N; ++i)
      ret[i] = ConvertFromFP8<T>(vals[i]);
    return ret;
  }
  // Intentionally public to allow access to the raw values.
  uint8_t vals[N];
};

template <size_t N> class fp8_e5m2_x {
  static constexpr size_t NExpBits = 5;
  static constexpr size_t NFracBits = 2;

  static_assert(N == 1 || N == 2,
                "fp8_e5m2_x: Template argument N must be 1 or 2");

  template <typename T> uint8_t ConvertToFP8(T h, saturation s) {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<std::decay_t<T>, bfloat16>)
      return s == saturation::finite
                 ? __builtin_spirv_ClampConvertBF16ToE5M2INTEL(h)
                 : __builtin_spirv_ConvertBF16ToE5M2EXT(h);
    const sycl::half halfValue = static_cast<sycl::half>(h);
    return s == saturation::finite
               ? __builtin_spirv_ClampConvertFP16ToE5M2INTEL(halfValue)
               : __builtin_spirv_ConvertFP16ToE5M2EXT(halfValue);
#else
    if constexpr (detail::IsSyclFpTypeV<T>) {
      return detail::ConvertFloatToFP8_CPU<NExpBits, NFracBits, T>(
          h, rounding::to_even, s);
    } else if constexpr (std::is_integral_v<std::decay_t<T>>) {
      return detail::ConvertIntToFP8_CPU<NExpBits, NFracBits, T>(
          h, rounding::to_even, s);
    }
#endif
  }

  template <typename T>
  void StochasticConvertToFP8(T h, uint32_t current_seed, uint32_t *pseed,
                              saturation s, uint8_t i) {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<T, bfloat16>) {
      if (s == saturation::finite) {
        vals[i] = __builtin_spirv_ClampStochasticRoundBF16ToE5M2INTEL(
            h, current_seed, pseed);
      } else {
        vals[i] = __builtin_spirv_StochasticRoundBF16ToE5M2INTEL(
            h, current_seed, pseed);
      }
    } else {
      if (s == saturation::finite) {
        vals[i] = __builtin_spirv_ClampStochasticRoundFP16ToE5M2INTEL(
            h, current_seed, pseed);
      } else {
        vals[i] = __builtin_spirv_StochasticRoundFP16ToE5M2INTEL(
            h, current_seed, pseed);
      }
    }
#endif
  }

  template <typename T>
  T ConvertFromFP8(uint8_t v, rounding r = rounding::to_even) const {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<std::decay_t<T>, bfloat16>)
      return __builtin_spirv_ConvertE5M2ToBF16EXT(v);
    sycl::half hi = __builtin_spirv_ConvertE5M2ToFP16EXT(v);
    return static_cast<T>(hi);
#else
    return detail::ConvertFromFP8ToBinaryFloat_CPU<NExpBits, NFracBits, T>(v,
                                                                           r);
#endif
  }

  void CheckConstraints(rounding r) const {
    assert(r == rounding::to_even &&
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
            typename = std::enable_if_t<detail::SyclfpVariadicV<N, Types...>>>
  explicit fp8_e5m2_x(Types... v) {
    using InT = std::common_type_t<std::decay_t<Types>...>;
    const InT in[N] = {v...};
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(in[i], saturation::finite);
  }

  template <typename T, typename = std::enable_if_t<detail::IsSyclFpTypeV<T>>>
  explicit fp8_e5m2_x(T const (&v)[N], rounding r = rounding::to_even,
                      saturation s = saturation::finite) {
    CheckConstraints(r);
    // TODO: optimize with vectorized builtin calls
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i], s);
  }

  template <typename T, typename = std::enable_if_t<detail::IsSyclFpTypeV<T>>>
  explicit fp8_e5m2_x(const sycl::marray<T, N> &v,
                      rounding r = rounding::to_even,
                      saturation s = saturation::finite) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = ConvertToFP8(v[i], s);
  }

  template <typename T,
            typename = std::enable_if_t<detail::IsOneOf<
                std::decay_t<T>, sycl::half, bfloat16, float>::value>>
  explicit fp8_e5m2_x([[maybe_unused]] T const (&in)[N],
                      [[maybe_unused]] const stochastic_seed &seed,
                      [[maybe_unused]] saturation s = saturation::finite) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    for (size_t i = 0; i < N; ++i) {
      StochasticConvertToFP8(in[i], current_seed, seed.pseed, s, i);
      current_seed = *seed.pseed;
    }
#endif
  }

  template <typename T,
            typename = std::enable_if_t<detail::IsOneOf<
                std::decay_t<T>, sycl::half, bfloat16, float>::value>>
  explicit fp8_e5m2_x([[maybe_unused]] const sycl::marray<T, N> &in,
                      [[maybe_unused]] const stochastic_seed &seed,
                      [[maybe_unused]] saturation s = saturation::finite) {

#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    for (size_t i = 0; i < N; ++i) {
      StochasticConvertToFP8(in[i], current_seed, seed.pseed, s, i);
      current_seed = *seed.pseed;
    }
#endif
  }

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && std::is_integral_v<T>>>
  explicit fp8_e5m2_x(T val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
  }

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && (detail::IsSyclFpTypeV<T> ||
                                                   std::is_integral_v<T>)>>
  fp8_e5m2_x &operator=(T val) {
    vals[0] = ConvertToFP8(val, saturation::finite);
    return *this;
  }

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && (detail::IsSyclFpTypeV<T> ||
                                                   std::is_integral_v<T>)>>
  explicit operator T() const {
    if constexpr (std::is_integral_v<T>)
      return ConvertFromFP8<T>(vals[0], rounding::toward_zero);
    else
      return ConvertFromFP8<T>(vals[0]);
  }

  // Convert to bool
  // Available only when N==1.

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bool() const {
    // false iff +0 or -0; otherwise true.
    return vals[0] != 0x00 && vals[0] != 0x80;
  }

  template <typename T, typename = std::enable_if_t<detail::IsOneOf<
                            std::decay_t<T>, sycl::half,
                            sycl::ext::oneapi::bfloat16, float>::value>>
  explicit operator sycl::marray<T, N>() const {
    sycl::marray<T, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] = ConvertFromFP8<T>(vals[i]);
    return out;
  }

  // Intentionally public to allow access to the raw values.

  uint8_t vals[N];
};

template <size_t N> class fp8_e8m0_x {
  static_assert(N == 1 || N == 2,
                "fp8_e8m0_x: Template argument N must be 1 or 2");

  void CheckConstraints(rounding r) const {
    assert((r == rounding::upward || r == rounding::toward_zero) &&
           "fp8_e8m0_x: only rounding::upward and rounding::toward_zero are "
           "supported");
  }

public:
  fp8_e8m0_x() = default;
  fp8_e8m0_x(const fp8_e8m0_x &) = default;
  ~fp8_e8m0_x() = default;
  fp8_e8m0_x &operator=(const fp8_e8m0_x &) = default;

  template <typename... Types,
            typename = std::enable_if_t<detail::SyclfpVariadicV<N, Types...>>>
  explicit fp8_e8m0_x(Types... v) {
    using InT = std::common_type_t<std::decay_t<Types>...>;
    const InT in[N] = {v...};
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertFloatToE8M0_CPU(in[i], rounding::upward,
                                               saturation::finite);
  }

  template <typename T, typename = std::enable_if_t<detail::IsSyclFpTypeV<T>>>
  explicit fp8_e8m0_x(T const (&in)[N], rounding r = rounding::upward) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertFloatToE8M0_CPU(in[i], r, saturation::finite);
  }

  template <typename T, typename = std::enable_if_t<detail::IsSyclFpTypeV<T>>>
  explicit fp8_e8m0_x(const marray<T, N> &in, rounding r = rounding::upward) {
    CheckConstraints(r);
    for (size_t i = 0; i < N; ++i)
      vals[i] = detail::ConvertFloatToE8M0_CPU(in[i], r, saturation::finite);
  }

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && std::is_integral_v<T>>>
  explicit fp8_e8m0_x(T val) {
    vals[0] =
        detail::ConvertIntToE8M0_CPU(val, rounding::upward, saturation::finite);
  }

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && (detail::IsSyclFpTypeV<T> ||
                                                   std::is_integral_v<T>)>>
  fp8_e8m0_x &operator=(T val) {
    if constexpr (std::is_integral_v<T>)
      vals[0] = detail::ConvertIntToE8M0_CPU(val, rounding::upward,
                                             saturation::finite);
    else
      vals[0] = detail::ConvertFloatToE8M0_CPU(val, rounding::upward,
                                               saturation::finite);
    return *this;
  }

  template <typename T, size_t M = N,
            typename = std::enable_if_t<M == 1 && (detail::IsSyclFpTypeV<T> ||
                                                   std::is_integral_v<T>)>>
  explicit operator T() const {
    if constexpr (std::is_integral_v<T>)
      return detail::ConvertFromE8M0_CPU<T>(vals[0], rounding::toward_zero);
    else
      return detail::ConvertFromE8M0_CPU<T>(vals[0], rounding::to_even);
  }

  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bool() const {
    return true;
  }

  template <typename T, typename = std::enable_if_t<detail::IsOneOf<
                            std::decay_t<T>, sycl::half,
                            sycl::ext::oneapi::bfloat16, float>::value>>
  explicit operator sycl::marray<T, N>() const {
    sycl::marray<T, N> out;
    for (size_t i = 0; i < N; ++i)
      out[i] = detail::ConvertFromE8M0_CPU<T>(vals[i], rounding::to_even);
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
