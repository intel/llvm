//==----------- types.hpp - sycl_ext_oneapi_fp4 ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>
#include <sycl/multi_ptr.hpp>

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/khr/static_addrspace_cast.hpp>
#include <sycl/marray.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#ifdef __SYCL_DEVICE_ONLY__

namespace sycl {
namespace detail {
using fp4_float16_vec2 = _Float16 __attribute__((ext_vector_type(2)));
using fp4_bfloat16_vec2 = __bf16 __attribute__((ext_vector_type(2)));
using fp4_uint8_vec2 = uint8_t __attribute__((ext_vector_type(2)));
} // namespace detail
} // namespace sycl

// FP4 builtins. The SPIR-V translator maps these names to the corresponding
// SPV_INTEL_float4 / SPV_INTEL_fp_conversions instructions. Scalar builtins
// take/return a 4-bit value held in an 8-bit register; the result is in the
// low 4 bits and the upper 4 bits are unused. Vec2 builtins operate on a
// pair of values; the encode-side returns a packed pair of nibbles in a
// single 8-bit value, while the decode-side accepts a vec2 of nibbles (each
// nibble in the low bits of its lane) and returns a vec2 of floats.

extern __DPCPP_SYCL_EXTERNAL uint8_t
__builtin_spirv_ClampConvertFP16ToE2M1INTEL(_Float16) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
    __builtin_spirv_ClampConvertFP16ToE2M1INTEL(
        ::sycl::detail::fp4_float16_vec2) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
__builtin_spirv_ClampConvertBF16ToE2M1INTEL(__bf16) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
    __builtin_spirv_ClampConvertBF16ToE2M1INTEL(
        ::sycl::detail::fp4_bfloat16_vec2) noexcept;

extern __DPCPP_SYCL_EXTERNAL _Float16
__builtin_spirv_ConvertE2M1ToFP16INTEL(uint8_t) noexcept;
extern __DPCPP_SYCL_EXTERNAL ::sycl::detail::fp4_float16_vec2
    __builtin_spirv_ConvertE2M1ToFP16INTEL(
        ::sycl::detail::fp4_uint8_vec2) noexcept;
extern __DPCPP_SYCL_EXTERNAL __bf16
__builtin_spirv_ConvertE2M1ToBF16INTEL(uint8_t) noexcept;
extern __DPCPP_SYCL_EXTERNAL ::sycl::detail::fp4_bfloat16_vec2
    __builtin_spirv_ConvertE2M1ToBF16INTEL(
        ::sycl::detail::fp4_uint8_vec2) noexcept;

extern __DPCPP_SYCL_EXTERNAL uint8_t
__builtin_spirv_StochasticRoundFP16ToE2M1INTEL(
    _Float16, uint32_t, __attribute__((opencl_private)) uint32_t *) noexcept;
extern __DPCPP_SYCL_EXTERNAL uint8_t
__builtin_spirv_StochasticRoundBF16ToE2M1INTEL(
    __bf16, uint32_t, __attribute__((opencl_private)) uint32_t *) noexcept;

#endif // __SYCL_DEVICE_ONLY__

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

enum class rounding {
  to_even,
  toward_zero,
};

struct stochastic_seed {
  explicit stochastic_seed(uint32_t *pseed) : pseed(pseed) {}
  uint32_t *const pseed;
};

namespace detail {

template <typename T> static inline int Fp4BitWidth(T x) noexcept {
  int width = 0;
  while (x != 0u) {
    ++width;
    x >>= 1;
  }
  return width;
}

template <typename T> struct Fp4SourceTraits;

template <> struct Fp4SourceTraits<float> {
  using UInt = uint32_t;
  static constexpr size_t ExpBits = 8;
  static constexpr size_t FracBits = 23;
  static constexpr int Bias = 127;
};

template <> struct Fp4SourceTraits<sycl::half> {
  using UInt = uint16_t;
  static constexpr size_t ExpBits = 5;
  static constexpr size_t FracBits = 10;
  static constexpr int Bias = 15;
};

template <> struct Fp4SourceTraits<sycl::ext::oneapi::bfloat16> {
  using UInt = uint16_t;
  static constexpr size_t ExpBits = 8;
  static constexpr size_t FracBits = 7;
  static constexpr int Bias = 127;
};

template <typename T> struct Fp4IntSourceTraits {
  using UnsignedT = std::make_unsigned_t<T>;
  static constexpr bool IsSigned = std::numeric_limits<T>::is_signed;
  static constexpr int ValueBits = std::numeric_limits<UnsignedT>::digits;
};

// E2M1 finite format constants.
struct FP4E2M1Traits {
  static constexpr int Ebits = 2;
  static constexpr int Mbits = 1;
  static constexpr uint8_t ExpAllOnes = 0x3u;
  static constexpr uint8_t MaxFrac = 0x1u;
  static constexpr int Bias = 1;
  static constexpr int Emin = 1 - Bias; // 0
  // E2M1 has no Inf and no NaN; all-ones exponent is the max normal.
  static constexpr uint8_t MaxFiniteExpField = ExpAllOnes;
  static constexpr uint8_t MaxFiniteFracField = MaxFrac;
  static constexpr uint8_t MaxFiniteCode =
      static_cast<uint8_t>((MaxFiniteExpField << Mbits) | MaxFiniteFracField);
  static constexpr int MaxFiniteExp =
      static_cast<int>(MaxFiniteExpField) - Bias; // 2
  static constexpr uint64_t MinNormalMantissa = uint64_t{1} << Mbits; // 2
  static constexpr uint64_t OverflowMantissa = uint64_t{1} << (Mbits + 1); // 4
  static constexpr uint64_t MaxFiniteMantissa =
      MinNormalMantissa + MaxFiniteFracField; // 3
};

// CPU host conversion: signed/unsigned integer to E2M1 nibble (low 4 bits).
template <typename T, typename Traits = Fp4IntSourceTraits<T>>
static inline uint8_t ConvertIntToFP4_CPU(T f, rounding R) noexcept {
  using UnsignedT = typename Traits::UnsignedT;
  using Format = FP4E2M1Traits;

  const uint8_t sign =
      (Traits::IsSigned && f < 0) ? static_cast<uint8_t>(0x8u) : 0u;
  UnsignedT magnitude = 0;

  if constexpr (Traits::IsSigned) {
    const UnsignedT bits = static_cast<UnsignedT>(f);
    magnitude = f < 0 ? static_cast<UnsignedT>(UnsignedT{0} - bits) : bits;
  } else {
    magnitude = static_cast<UnsignedT>(f);
  }

  if (magnitude == 0)
    return sign;

  int unbiasedExp = Fp4BitWidth(static_cast<uint64_t>(magnitude)) - 1;
  if (unbiasedExp > Format::MaxFiniteExp)
    return static_cast<uint8_t>(sign | Format::MaxFiniteCode);

  const int shift = unbiasedExp - Format::Mbits;
  uint64_t mantissa = 0u;
  if (shift <= 0) {
    mantissa = static_cast<uint64_t>(magnitude) << (-shift);
  } else {
    const uint64_t truncated = static_cast<uint64_t>(magnitude) >> shift;
    const uint64_t remainderMask = (uint64_t{1} << shift) - 1u;
    const uint64_t remainder =
        static_cast<uint64_t>(magnitude) & remainderMask;

    mantissa = truncated;
    if (remainder != 0u && R == rounding::to_even) {
      const uint64_t half = uint64_t{1} << (shift - 1);
      if (remainder > half ||
          (remainder == half && (truncated & uint64_t{1}) != 0u))
        ++mantissa;
    }
  }

  if (mantissa >= Format::OverflowMantissa) {
    mantissa = Format::MinNormalMantissa;
    ++unbiasedExp;
  }

  if (unbiasedExp > Format::MaxFiniteExp ||
      (unbiasedExp == Format::MaxFiniteExp &&
       mantissa > Format::MaxFiniteMantissa))
    return static_cast<uint8_t>(sign | Format::MaxFiniteCode);

  const uint8_t expField =
      static_cast<uint8_t>(unbiasedExp + Format::Bias);
  const uint8_t fracField =
      static_cast<uint8_t>(mantissa - Format::MinNormalMantissa);
  return static_cast<uint8_t>(
      sign | static_cast<uint8_t>(expField << Format::Mbits) | fracField);
}

// CPU host conversion: binary floating point to E2M1 nibble (low 4 bits).
template <typename T, typename Traits = Fp4SourceTraits<T>>
static inline uint8_t ConvertFloatToFP4_CPU(T f, rounding R) noexcept {
  using UInt = typename Traits::UInt;
  using Format = FP4E2M1Traits;

  constexpr UInt SignMask = UInt{1} << (Traits::ExpBits + Traits::FracBits);
  constexpr UInt FracMask = (UInt{1} << Traits::FracBits) - UInt{1};
  constexpr UInt ExpMask = ((UInt{1} << Traits::ExpBits) - UInt{1})
                           << Traits::FracBits;
  constexpr UInt ExpAllOnes = (UInt{1} << Traits::ExpBits) - UInt{1};

  UInt bits;
  std::memcpy(&bits, &f, sizeof(bits));

  const uint8_t sign = (bits & SignMask) ? 0x8u : 0x0u;
  bits &= ~SignMask;

  const UInt exp = (bits & ExpMask) >> Traits::FracBits;
  const UInt frac = bits & FracMask;

  // Inf and NaN both clamp to max normal preserving sign (E2M1 has neither).
  if (exp == ExpAllOnes)
    return static_cast<uint8_t>(sign | Format::MaxFiniteCode);

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
    if (shift >= 64)
      return 0u;

    const uint64_t truncated = value >> shift;
    const uint64_t remainderMask = (uint64_t{1} << shift) - 1u;
    const uint64_t remainder = value & remainderMask;

    if (remainder == 0u || R == rounding::toward_zero)
      return truncated;

    const uint64_t half = uint64_t{1} << (shift - 1);
    if (remainder > half)
      return truncated + 1u;
    if (remainder < half)
      return truncated;
    return (truncated & 1u) != 0u ? truncated + 1u : truncated;
  };

  if (unbiasedExp > Format::MaxFiniteExp)
    return static_cast<uint8_t>(sign | Format::MaxFiniteCode);

  if (unbiasedExp == Format::MaxFiniteExp) {
    const uint64_t lhs = significand << Format::Mbits;
    const uint64_t rhs = Format::MaxFiniteMantissa << leadingBit;
    if (lhs > rhs)
      return static_cast<uint8_t>(sign | Format::MaxFiniteCode);
  }

  if (unbiasedExp < Format::Emin) {
    const int shift =
        leadingBit - unbiasedExp - Format::Bias - Format::Mbits + 1;
    uint64_t mantissa = shift > 0 ? roundShiftRight(significand, shift)
                                  : (significand << (-shift));

    if (mantissa == 0u)
      return sign;

    if (mantissa >= Format::MinNormalMantissa)
      return static_cast<uint8_t>(sign |
                                  (uint8_t{1} << Format::Mbits));

    return static_cast<uint8_t>(sign | static_cast<uint8_t>(mantissa));
  }

  const int shift = leadingBit - Format::Mbits;
  uint64_t mantissa = shift > 0 ? roundShiftRight(significand, shift)
                                : (significand << (-shift));

  if (mantissa >= Format::OverflowMantissa) {
    mantissa = Format::MinNormalMantissa;
    ++unbiasedExp;
  }

  if (unbiasedExp > Format::MaxFiniteExp ||
      (unbiasedExp == Format::MaxFiniteExp &&
       mantissa > Format::MaxFiniteMantissa))
    return static_cast<uint8_t>(sign | Format::MaxFiniteCode);

  const uint8_t expField =
      static_cast<uint8_t>(unbiasedExp + Format::Bias);
  const uint8_t fracField =
      static_cast<uint8_t>(mantissa - Format::MinNormalMantissa);
  return static_cast<uint8_t>(
      sign | static_cast<uint8_t>(expField << Format::Mbits) | fracField);
}

template <typename ToT, typename = void>
struct Fp4HasFloatTraits : std::false_type {};
template <typename ToT>
struct Fp4HasFloatTraits<ToT,
                         std::void_t<decltype(Fp4SourceTraits<ToT>::ExpBits),
                                     decltype(Fp4SourceTraits<ToT>::FracBits),
                                     decltype(Fp4SourceTraits<ToT>::Bias)>>
    : std::true_type {};

// CPU host conversion: E2M1 nibble (low 4 bits of `code`) to ToT.
template <typename ToT>
static inline ToT ConvertFromFP4ToBinaryFloat_CPU(uint8_t code,
                                                  rounding R) noexcept {
  using Format = FP4E2M1Traits;
  constexpr uint8_t SignBit = 0x8u;
  constexpr uint8_t ExpAllOnes = Format::ExpAllOnes;
  constexpr uint8_t FracMask = Format::MaxFrac;

  const bool negative = (code & SignBit) != 0u;
  const uint8_t exp = static_cast<uint8_t>((code >> Format::Mbits) & ExpAllOnes);
  const uint8_t frac = static_cast<uint8_t>(code & FracMask);

  uint32_t significand = 0u;
  int exp2 = 0;

  if (exp == 0u) {
    if (frac == 0u) {
      significand = 0u;
    } else {
      significand = frac;
      exp2 = Format::Emin;
    }
  } else {
    significand =
        static_cast<uint32_t>((1u << Format::Mbits) + frac);
    exp2 = static_cast<int>(exp) - Format::Bias;
  }

  if constexpr (Fp4HasFloatTraits<ToT>::value) {
    using Traits = Fp4SourceTraits<ToT>;
    using UInt = typename Traits::UInt;

    constexpr UInt ExpAllOnesDst = ((UInt{1} << Traits::ExpBits) - UInt{1})
                                   << Traits::FracBits;
    constexpr UInt FracMaskDst = (UInt{1} << Traits::FracBits) - UInt{1};

    UInt bits = 0;
    if (significand == 0u) {
      bits =
          negative ? (UInt{1} << (Traits::ExpBits + Traits::FracBits)) : 0u;
    } else {
      const int sigBits = Fp4BitWidth(significand);
      const int unbiasedExp = exp2 + sigBits - 1 - Format::Mbits;
      const UInt signBit =
          negative ? (UInt{1} << (Traits::ExpBits + Traits::FracBits)) : 0u;

      const int shift = static_cast<int>(Traits::FracBits) - (sigBits - 1);
      const UInt aligned = static_cast<UInt>(significand) << shift;
      const UInt expField = static_cast<UInt>(unbiasedExp + Traits::Bias)
                            << Traits::FracBits;
      bits = signBit | expField | (aligned & FracMaskDst);
    }

    (void)R;
    (void)ExpAllOnesDst;
    return __builtin_bit_cast(ToT, bits);
  } else if constexpr (std::is_integral_v<ToT>) {
    using Traits = Fp4IntSourceTraits<ToT>;
    using UnsignedT = typename Traits::UnsignedT;

    if (significand == 0u)
      return ToT{};

    const int shift = exp2 - Format::Mbits;
    uint64_t magnitude = 0u;

    if (shift >= 0) {
      magnitude = static_cast<uint64_t>(significand) << shift;
    } else {
      const int rshift = -shift;
      magnitude = static_cast<uint64_t>(significand) >> rshift;
      // rounding::toward_zero: discard remainder bits.
      (void)R;
    }

    if (magnitude == 0u)
      return ToT{};

    if (Fp4BitWidth(magnitude) > Traits::ValueBits) {
      if constexpr (Traits::IsSigned)
        return negative ? std::numeric_limits<ToT>::min()
                        : std::numeric_limits<ToT>::max();
      else
        return negative ? ToT{0} : std::numeric_limits<ToT>::max();
    }

    const UnsignedT narrowed = static_cast<UnsignedT>(magnitude);
    if constexpr (Traits::IsSigned)
      return static_cast<ToT>(negative ? -static_cast<ToT>(narrowed)
                                       : static_cast<ToT>(narrowed));
    return static_cast<ToT>(narrowed);
  } else {
    (void)R;
    return ToT{};
  }
}

// Pack two nibbles into a single byte: lo in low 4 bits, hi in high 4 bits.
static inline uint8_t Fp4Pack(uint8_t lo, uint8_t hi) noexcept {
  return static_cast<uint8_t>((lo & 0x0Fu) | ((hi & 0x0Fu) << 4));
}

// Extract element i (0 or 1) from packed byte.
static inline uint8_t Fp4Extract(uint8_t packed, size_t i) noexcept {
  return static_cast<uint8_t>((packed >> (i * 4)) & 0x0Fu);
}

} // namespace detail

template <size_t N> class fp4_e2m1_x {
  static_assert(N == 1 || N == 2,
                "fp4_e2m1_x: Template argument N must be 1 or 2");

  template <typename T,
            typename = std::enable_if_t<std::is_integral_v<std::decay_t<T>>>>
  uint8_t ConvertToFP4(T h) {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<std::decay_t<T>, char> ||
                  std::is_same_v<std::decay_t<T>, signed char> ||
                  std::is_same_v<std::decay_t<T>, unsigned char>) {
      const _Float16 v = static_cast<_Float16>(h);
      return __builtin_spirv_ClampConvertFP16ToE2M1INTEL(v);
    }
    return detail::ConvertIntToFP4_CPU<T>(h, rounding::to_even);
#else
    return detail::ConvertIntToFP4_CPU<T>(h, rounding::to_even);
#endif
  }

  uint8_t ConvertToFP4(sycl::half h) {
#ifdef __SYCL_DEVICE_ONLY__
    const _Float16 v = sycl::bit_cast<_Float16>(h);
    return __builtin_spirv_ClampConvertFP16ToE2M1INTEL(v);
#else
    return detail::ConvertFloatToFP4_CPU<sycl::half>(h, rounding::to_even);
#endif
  }

#ifdef __SYCL_DEVICE_ONLY__
  uint8_t ConvertToFP4_Vec2(::sycl::detail::fp4_float16_vec2 h) {
    return __builtin_spirv_ClampConvertFP16ToE2M1INTEL(h);
  }
#endif

  uint8_t ConvertToFP4(float h) {
    return detail::ConvertFloatToFP4_CPU<float>(h, rounding::to_even);
  }

#ifdef __SYCL_DEVICE_ONLY__
  uint8_t ConvertBF16ToFP4_Vec2(::sycl::detail::fp4_bfloat16_vec2 h) {
    return __builtin_spirv_ClampConvertBF16ToE2M1INTEL(h);
  }
#endif

  uint8_t ConvertBF16ToFP4(bfloat16 h) {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_spirv_ClampConvertBF16ToE2M1INTEL(
        sycl::bit_cast<__bf16>(h));
#else
    return detail::ConvertFloatToFP4_CPU<bfloat16>(h, rounding::to_even);
#endif
  }

  template <typename T> T ConvertFromFP4(uint8_t v) const {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::half hi = __builtin_spirv_ConvertE2M1ToFP16INTEL(v);
    return static_cast<T>(hi);
#else
    return detail::ConvertFromFP4ToBinaryFloat_CPU<T>(v, rounding::toward_zero);
#endif
  }

  template <typename T> T ConvertFromFP4Int(uint8_t v) const {
    return detail::ConvertFromFP4ToBinaryFloat_CPU<T>(v, rounding::toward_zero);
  }

  void ConvertFromFP4_Vec2(sycl::marray<sycl::half, N> &ret) const {
#ifdef __SYCL_DEVICE_ONLY__
    const ::sycl::detail::fp4_uint8_vec2 packed{
        detail::Fp4Extract(vals[0], 0), detail::Fp4Extract(vals[0], 1)};
    ::sycl::detail::fp4_float16_vec2 hi =
        __builtin_spirv_ConvertE2M1ToFP16INTEL(packed);
    ret[0] = sycl::bit_cast<sycl::half>(hi[0]);
    ret[1] = sycl::bit_cast<sycl::half>(hi[1]);
#else
    for (size_t i = 0; i < 2; ++i)
      ret[i] = detail::ConvertFromFP4ToBinaryFloat_CPU<sycl::half>(
          detail::Fp4Extract(vals[0], i), rounding::toward_zero);
#endif
  }

  bfloat16 ConvertBF16FromFP4(uint8_t v) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::bit_cast<bfloat16>(__builtin_spirv_ConvertE2M1ToBF16INTEL(v));
#else
    return detail::ConvertFromFP4ToBinaryFloat_CPU<bfloat16>(
        v, rounding::toward_zero);
#endif
  }

  void ConvertBF16FromFP4_Vec2(sycl::marray<bfloat16, N> &ret) const {
#ifdef __SYCL_DEVICE_ONLY__
    const ::sycl::detail::fp4_uint8_vec2 packed{
        detail::Fp4Extract(vals[0], 0), detail::Fp4Extract(vals[0], 1)};
    ::sycl::detail::fp4_bfloat16_vec2 hi =
        __builtin_spirv_ConvertE2M1ToBF16INTEL(packed);
    ret[0] = sycl::bit_cast<bfloat16>(hi[0]);
    ret[1] = sycl::bit_cast<bfloat16>(hi[1]);
#else
    for (size_t i = 0; i < 2; ++i)
      ret[i] = detail::ConvertFromFP4ToBinaryFloat_CPU<bfloat16>(
          detail::Fp4Extract(vals[0], i), rounding::toward_zero);
#endif
  }

  void CheckConstraints(rounding r) const {
    assert(r == rounding::to_even &&
           "fp4_e2m1_x: only rounding::to_even is supported");
  }

  // Store one nibble at element index i (0 or 1).
  void StoreNibble(size_t i, uint8_t nibble) {
    if (i == 0)
      vals[0] = static_cast<uint8_t>((vals[0] & 0xF0u) | (nibble & 0x0Fu));
    else
      vals[0] = static_cast<uint8_t>((vals[0] & 0x0Fu) |
                                     (static_cast<uint8_t>(nibble & 0x0Fu)
                                      << 4));
  }

#ifdef __SYCL_DEVICE_ONLY__
#define CONVERT_TO_FP4(VecType, CastType, in, Prefix)                          \
  if constexpr (N == 1) {                                                      \
    vals[0] = Convert##Prefix##ToFP4(in[0]);                                   \
  } else {                                                                     \
    const VecType vec{sycl::bit_cast<CastType>(in[0]),                         \
                      sycl::bit_cast<CastType>(in[1])};                        \
    vals[0] = Convert##Prefix##ToFP4_Vec2(vec);                                \
  }
#else
#define CONVERT_TO_FP4(VecType, CastType, in, Prefix)                          \
  if constexpr (N == 1) {                                                      \
    vals[0] = Convert##Prefix##ToFP4(in[0]);                                   \
  } else {                                                                     \
    const uint8_t lo = Convert##Prefix##ToFP4(in[0]);                          \
    const uint8_t hi = Convert##Prefix##ToFP4(in[1]);                          \
    vals[0] = detail::Fp4Pack(lo, hi);                                         \
  }
#endif

public:
  fp4_e2m1_x() = default;
  fp4_e2m1_x(const fp4_e2m1_x &) = default;
  ~fp4_e2m1_x() = default;
  fp4_e2m1_x &operator=(const fp4_e2m1_x &) = default;

  // Construct from pack of half, bfloat16, float.
  // Available only when the size of the pack is equal to N.
  template <typename... Types,
            typename = std::enable_if_t<
                (sizeof...(Types) == N) &&
                (((std::is_same_v<std::decay_t<Types>, half>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, bfloat16>) && ...) ||
                 ((std::is_same_v<std::decay_t<Types>, float>) && ...))>>
  explicit fp4_e2m1_x(Types... v) {
    if constexpr (((std::is_same_v<std::decay_t<Types>, bfloat16>) && ...)) {
      const bfloat16 in[N] = {v...};
      CONVERT_TO_FP4(::sycl::detail::fp4_bfloat16_vec2, __bf16, in, BF16);
    } else if constexpr (((std::is_same_v<std::decay_t<Types>, half>) && ...)) {
      const sycl::half in[N] = {v...};
      CONVERT_TO_FP4(::sycl::detail::fp4_float16_vec2, _Float16, in, );
    } else {
      const float in[N] = {v...};
      if constexpr (N == 1) {
        vals[0] = ConvertToFP4(in[0]);
      } else {
        const uint8_t lo = ConvertToFP4(in[0]);
        const uint8_t hi = ConvertToFP4(in[1]);
        vals[0] = detail::Fp4Pack(lo, hi);
      }
    }
  }

  // Construct from an array of half, bfloat16, float.
  explicit fp4_e2m1_x(sycl::half const (&v)[N],
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    CONVERT_TO_FP4(::sycl::detail::fp4_float16_vec2, _Float16, v, );
  }

  explicit fp4_e2m1_x(bfloat16 const (&v)[N], rounding r = rounding::to_even) {
    CheckConstraints(r);
    CONVERT_TO_FP4(::sycl::detail::fp4_bfloat16_vec2, __bf16, v, BF16);
  }

  explicit fp4_e2m1_x(float const (&v)[N], rounding r = rounding::to_even) {
    CheckConstraints(r);
    if constexpr (N == 1) {
      vals[0] = ConvertToFP4(v[0]);
    } else {
      const uint8_t lo = ConvertToFP4(v[0]);
      const uint8_t hi = ConvertToFP4(v[1]);
      vals[0] = detail::Fp4Pack(lo, hi);
    }
  }

  // Construct from an marray of half, bfloat16, float.
  explicit fp4_e2m1_x(const sycl::marray<sycl::half, N> &v,
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    CONVERT_TO_FP4(::sycl::detail::fp4_float16_vec2, _Float16, v, );
  }

  explicit fp4_e2m1_x(const sycl::marray<bfloat16, N> &v,
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    CONVERT_TO_FP4(::sycl::detail::fp4_bfloat16_vec2, __bf16, v, BF16);
  }

  explicit fp4_e2m1_x(const sycl::marray<float, N> &v,
                      rounding r = rounding::to_even) {
    CheckConstraints(r);
    if constexpr (N == 1) {
      vals[0] = ConvertToFP4(v[0]);
    } else {
      const uint8_t lo = ConvertToFP4(v[0]);
      const uint8_t hi = ConvertToFP4(v[1]);
      vals[0] = detail::Fp4Pack(lo, hi);
    }
  }

  // Construct with stochastic rounding from an array of half, bfloat16.
  explicit fp4_e2m1_x([[maybe_unused]] half const (&in)[N],
                      [[maybe_unused]] const stochastic_seed &seed) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    uint32_t next_seed = 0;
    uint8_t nibbles[2] = {0, 0};
    for (size_t i = 0; i < N; ++i) {
      const _Float16 v = sycl::bit_cast<_Float16>(in[i]);
      nibbles[i] = __builtin_spirv_StochasticRoundFP16ToE2M1INTEL(
          v, current_seed,
          sycl::khr::static_addrspace_cast<
              sycl::access::address_space::private_space>(&next_seed)
              .get_decorated());
      current_seed = next_seed;
      next_seed = 0;
    }
    if constexpr (N == 1)
      vals[0] = static_cast<uint8_t>(nibbles[0] & 0x0Fu);
    else
      vals[0] = detail::Fp4Pack(nibbles[0], nibbles[1]);
#else
    throw std::runtime_error(
        "stochastic rounding constructors are not supported on host");
#endif
  }

  explicit fp4_e2m1_x([[maybe_unused]] bfloat16 const (&in)[N],
                      [[maybe_unused]] const stochastic_seed &seed) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    uint32_t next_seed = 0;
    uint8_t nibbles[2] = {0, 0};
    for (size_t i = 0; i < N; ++i) {
      nibbles[i] = __builtin_spirv_StochasticRoundBF16ToE2M1INTEL(
          sycl::bit_cast<__bf16>(in[i]), current_seed,
          sycl::khr::static_addrspace_cast<
              sycl::access::address_space::private_space>(&next_seed)
              .get_decorated());
      current_seed = next_seed;
      next_seed = 0;
    }
    if constexpr (N == 1)
      vals[0] = static_cast<uint8_t>(nibbles[0] & 0x0Fu);
    else
      vals[0] = detail::Fp4Pack(nibbles[0], nibbles[1]);
#else
    throw std::runtime_error(
        "stochastic rounding constructors are not supported on host");
#endif
  }

  // Construct with stochastic rounding from an marray of half, bfloat16.
  explicit fp4_e2m1_x([[maybe_unused]] const sycl::marray<half, N> &in,
                      [[maybe_unused]] const stochastic_seed &seed) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    uint32_t next_seed = 0;
    uint8_t nibbles[2] = {0, 0};
    for (size_t i = 0; i < N; ++i) {
      const _Float16 v = sycl::bit_cast<_Float16>(in[i]);
      nibbles[i] = __builtin_spirv_StochasticRoundFP16ToE2M1INTEL(
          v, current_seed,
          sycl::khr::static_addrspace_cast<
              sycl::access::address_space::private_space>(&next_seed)
              .get_decorated());
      current_seed = next_seed;
      next_seed = 0;
    }
    if constexpr (N == 1)
      vals[0] = static_cast<uint8_t>(nibbles[0] & 0x0Fu);
    else
      vals[0] = detail::Fp4Pack(nibbles[0], nibbles[1]);
#else
    throw std::runtime_error(
        "stochastic rounding constructors are not supported on host");
#endif
  }

  explicit fp4_e2m1_x([[maybe_unused]] const sycl::marray<bfloat16, N> &in,
                      [[maybe_unused]] const stochastic_seed &seed) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t current_seed = *seed.pseed;
    uint32_t next_seed = 0;
    uint8_t nibbles[2] = {0, 0};
    for (size_t i = 0; i < N; ++i) {
      nibbles[i] = __builtin_spirv_StochasticRoundBF16ToE2M1INTEL(
          sycl::bit_cast<__bf16>(in[i]), current_seed,
          sycl::khr::static_addrspace_cast<
              sycl::access::address_space::private_space>(&next_seed)
              .get_decorated());
      current_seed = next_seed;
      next_seed = 0;
    }
    if constexpr (N == 1)
      vals[0] = static_cast<uint8_t>(nibbles[0] & 0x0Fu);
    else
      vals[0] = detail::Fp4Pack(nibbles[0], nibbles[1]);
#else
    throw std::runtime_error(
        "stochastic rounding constructors are not supported on host");
#endif
  }

  // Construct from integer types. Available only when N==1.
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp4_e2m1_x(short val) {
    vals[0] = ConvertToFP4(val);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp4_e2m1_x(int val) {
    vals[0] = ConvertToFP4(val);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp4_e2m1_x(long val) {
    vals[0] = ConvertToFP4(val);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp4_e2m1_x(long long val) {
    vals[0] = ConvertToFP4(val);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp4_e2m1_x(unsigned short val) {
    vals[0] = ConvertToFP4(val);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp4_e2m1_x(unsigned int val) {
    vals[0] = ConvertToFP4(val);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp4_e2m1_x(unsigned long val) {
    vals[0] = ConvertToFP4(val);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit fp4_e2m1_x(unsigned long long val) {
    vals[0] = ConvertToFP4(val);
  }

  // Assign (operator) from half, bfloat16, float, and integer types.
  // Available only when N==1.
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(sycl::half val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(bfloat16 val) {
    vals[0] = ConvertBF16ToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(float val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(short val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(int val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(long val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(long long val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(unsigned short val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(unsigned int val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(unsigned long val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  fp4_e2m1_x &operator=(unsigned long long val) {
    vals[0] = ConvertToFP4(val);
    return *this;
  }

  // Convert to half, bfloat16, float. Available only when N==1.
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator half() const {
    return ConvertFromFP4<sycl::half>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bfloat16() const {
    return ConvertBF16FromFP4(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator float() const {
    return ConvertFromFP4<float>(vals[0] & 0x0Fu);
  }

  // Convert to integer types. Available only when N==1.
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator char() const {
    return ConvertFromFP4Int<char>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator signed char() const {
    return ConvertFromFP4Int<signed char>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator short() const {
    return ConvertFromFP4Int<short>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator int() const {
    return ConvertFromFP4Int<int>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator long() const {
    return ConvertFromFP4Int<long>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator long long() const {
    return ConvertFromFP4Int<long long>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned char() const {
    return ConvertFromFP4Int<unsigned char>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned short() const {
    return ConvertFromFP4Int<unsigned short>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned int() const {
    return ConvertFromFP4Int<unsigned int>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned long() const {
    return ConvertFromFP4Int<unsigned long>(vals[0] & 0x0Fu);
  }
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator unsigned long long() const {
    return ConvertFromFP4Int<unsigned long long>(vals[0] & 0x0Fu);
  }

  // Convert to bool. Available only when N==1.
  // false iff +0 or -0; otherwise true.
  template <size_t M = N, typename = std::enable_if_t<M == 1>>
  explicit operator bool() const {
    const uint8_t low = vals[0] & 0x0Fu;
    return low != 0x0u && low != 0x8u;
  }

  // Convert to marray of half, bfloat16, float.
  explicit operator sycl::marray<sycl::half, N>() const {
    sycl::marray<sycl::half, N> ret;
    if constexpr (N == 1)
      ret[0] = ConvertFromFP4<sycl::half>(vals[0] & 0x0Fu);
    else
      ConvertFromFP4_Vec2(ret);
    return ret;
  }

  explicit operator sycl::marray<bfloat16, N>() const {
    sycl::marray<bfloat16, N> ret;
    if constexpr (N == 1)
      ret[0] = ConvertBF16FromFP4(vals[0] & 0x0Fu);
    else
      ConvertBF16FromFP4_Vec2(ret);
    return ret;
  }

  explicit operator sycl::marray<float, N>() const {
    sycl::marray<float, N> ret;
    for (size_t i = 0; i < N; ++i)
      ret[i] = detail::ConvertFromFP4ToBinaryFloat_CPU<float>(
          detail::Fp4Extract(vals[0], i), rounding::toward_zero);
    return ret;
  }

  // Intentionally public to allow access to the raw values.
  // Element 0 is in the low 4 bits of vals[0].
  // Element 1 (if it exists) is in the high 4 bits of vals[0].
  uint8_t vals[(N + 1) / 2];
#undef CONVERT_TO_FP4
};

// Deduction guide. Available only when the size of the pack is greater than
// zero.
template <typename... Ts> fp4_e2m1_x(Ts...) -> fp4_e2m1_x<sizeof...(Ts)>;

using fp4_e2m1 = fp4_e2m1_x<1>;
using fp4_e2m1_x2 = fp4_e2m1_x<2>;

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
