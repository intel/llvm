//==--------- bfloat16.hpp ------- SYCL bfloat16 conversion ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aliases.hpp>                   // for half
#include <sycl/detail/defines_elementary.hpp> // for __DPCPP_SYCL_EXTERNAL
#include <sycl/half_type.hpp>                 // for half

#include <stdint.h> // for uint16_t, uint32_t

extern "C" __DPCPP_SYCL_EXTERNAL uint16_t
__devicelib_ConvertFToBF16INTEL(const float &) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL float
__devicelib_ConvertBF16ToFINTEL(const uint16_t &) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertFToBF16INTELVec1(const float *, uint16_t *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertBF16ToFINTELVec1(const uint16_t *, float *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertFToBF16INTELVec2(const float *, uint16_t *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertBF16ToFINTELVec2(const uint16_t *, float *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertFToBF16INTELVec3(const float *, uint16_t *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertBF16ToFINTELVec3(const uint16_t *, float *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertFToBF16INTELVec4(const float *, uint16_t *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertBF16ToFINTELVec4(const uint16_t *, float *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertFToBF16INTELVec8(const float *, uint16_t *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertBF16ToFINTELVec8(const uint16_t *, float *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertFToBF16INTELVec16(const float *, uint16_t *) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL void
__devicelib_ConvertBF16ToFINTELVec16(const uint16_t *, float *) noexcept;

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {

class bfloat16;

namespace detail {
using Bfloat16StorageT = uint16_t;
Bfloat16StorageT bfloat16ToBits(const bfloat16 &Value);
bfloat16 bitsToBfloat16(const Bfloat16StorageT Value);
// Class to convert different data types to Bfloat16
// with different rounding modes.
class ConvertToBfloat16;

template <int N> void BF16VecToFloatVec(const bfloat16 src[N], float dst[N]) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
  const uint16_t *src_i16 = sycl::bit_cast<const uint16_t *>(src);
  if constexpr (N == 1)
    __devicelib_ConvertBF16ToFINTELVec1(src_i16, dst);
  else if constexpr (N == 2)
    __devicelib_ConvertBF16ToFINTELVec2(src_i16, dst);
  else if constexpr (N == 3)
    __devicelib_ConvertBF16ToFINTELVec3(src_i16, dst);
  else if constexpr (N == 4)
    __devicelib_ConvertBF16ToFINTELVec4(src_i16, dst);
  else if constexpr (N == 8)
    __devicelib_ConvertBF16ToFINTELVec8(src_i16, dst);
  else if constexpr (N == 16)
    __devicelib_ConvertBF16ToFINTELVec16(src_i16, dst);
#else
  for (int i = 0; i < N; ++i) {
    dst[i] = (float)src[i];
  }
#endif
}

// sycl::vec support
namespace bf16 {
#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
#ifdef __SYCL_DEVICE_ONLY__
using Vec2StorageT = Bfloat16StorageT __attribute__((ext_vector_type(2)));
using Vec3StorageT = Bfloat16StorageT __attribute__((ext_vector_type(3)));
using Vec4StorageT = Bfloat16StorageT __attribute__((ext_vector_type(4)));
using Vec8StorageT = Bfloat16StorageT __attribute__((ext_vector_type(8)));
using Vec16StorageT = Bfloat16StorageT __attribute__((ext_vector_type(16)));
#else
using Vec2StorageT = std::array<Bfloat16StorageT, 2>;
using Vec3StorageT = std::array<Bfloat16StorageT, 3>;
using Vec4StorageT = std::array<Bfloat16StorageT, 4>;
using Vec8StorageT = std::array<Bfloat16StorageT, 8>;
using Vec16StorageT = std::array<Bfloat16StorageT, 16>;
#endif
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
} // namespace bf16
} // namespace detail

class bfloat16 {
protected:
  detail::Bfloat16StorageT value;

  friend inline detail::Bfloat16StorageT
  detail::bfloat16ToBits(const bfloat16 &Value);
  friend inline bfloat16
  detail::bitsToBfloat16(const detail::Bfloat16StorageT Value);
  friend class detail::ConvertToBfloat16;

public:
  bfloat16() = default;
  constexpr bfloat16(const bfloat16 &) = default;
  constexpr bfloat16(bfloat16 &&) = default;
  constexpr bfloat16 &operator=(const bfloat16 &rhs) = default;
  ~bfloat16() = default;

private:
  static detail::Bfloat16StorageT from_float_fallback(const float &a) {
    // We don't call sycl::isnan because we don't want a data type to depend on
    // builtins.
    if (a != a)
      return 0xffc1;

    union {
      uint32_t intStorage;
      float floatValue;
    };
    floatValue = a;
    // Do RNE and truncate
    uint32_t roundingBias = ((intStorage >> 16) & 0x1) + 0x00007FFF;
    return static_cast<uint16_t>((intStorage + roundingBias) >> 16);
  }

  // Explicit conversion functions
  static detail::Bfloat16StorageT from_float(const float &a) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
#if (__SYCL_CUDA_ARCH__ >= 800)
    detail::Bfloat16StorageT res;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(res) : "f"(a));
    return res;
#else
    return from_float_fallback(a);
#endif
#elif defined(__AMDGCN__)
    return from_float_fallback(a);
#else
    return __devicelib_ConvertFToBF16INTEL(a);
#endif
#endif
    return from_float_fallback(a);
  }

  static float to_float(const detail::Bfloat16StorageT &a) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
    return __devicelib_ConvertBF16ToFINTEL(a);
#else
    union {
      uint32_t intStorage;
      float floatValue;
    };
    intStorage = a << 16;
    return floatValue;
#endif
  }

protected:
  friend class sycl::vec<bfloat16, 1>;
  friend class sycl::vec<bfloat16, 2>;
  friend class sycl::vec<bfloat16, 3>;
  friend class sycl::vec<bfloat16, 4>;
  friend class sycl::vec<bfloat16, 8>;
  friend class sycl::vec<bfloat16, 16>;

public:
  // Implicit conversion from float to bfloat16
  bfloat16(const float &a) { value = from_float(a); }

  bfloat16 &operator=(const float &rhs) {
    value = from_float(rhs);
    return *this;
  }

  // Implicit conversion from sycl::half to bfloat16
  bfloat16(const sycl::half &a) { value = from_float(a); }

  bfloat16 &operator=(const sycl::half &rhs) {
    value = from_float(rhs);
    return *this;
  }

  // Implicit conversion from bfloat16 to float
  operator float() const { return to_float(value); }

  // Implicit conversion from bfloat16 to sycl::half
  operator sycl::half() const { return to_float(value); }

  // Logical operators (!,||,&&) are covered if we can cast to bool
  explicit operator bool() { return to_float(value) != 0.0f; }

  // Unary minus operator overloading
  friend bfloat16 operator-(bfloat16 &lhs) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    (__SYCL_CUDA_ARCH__ >= 800)
    detail::Bfloat16StorageT res;
    asm("neg.bf16 %0, %1;" : "=h"(res) : "h"(lhs.value));
    return detail::bitsToBfloat16(res);
#elif defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
    return bfloat16{-__devicelib_ConvertBF16ToFINTEL(lhs.value)};
#else
    return bfloat16{-to_float(lhs.value)};
#endif
  }

// Increment and decrement operators overloading
#define OP(op)                                                                 \
  friend bfloat16 &operator op(bfloat16 & lhs) {                               \
    float f = to_float(lhs.value);                                             \
    lhs.value = from_float(op f);                                              \
    return lhs;                                                                \
  }                                                                            \
  friend bfloat16 operator op(bfloat16 &lhs, int) {                            \
    bfloat16 old = lhs;                                                        \
    operator op(lhs);                                                          \
    return old;                                                                \
  }
  OP(++)
  OP(--)
#undef OP

  // Assignment operators overloading
#define OP(op)                                                                 \
  friend bfloat16 &operator op(bfloat16 & lhs, const bfloat16 & rhs) {         \
    float f = static_cast<float>(lhs);                                         \
    f op static_cast<float>(rhs);                                              \
    return lhs = f;                                                            \
  }
  OP(+=)
  OP(-=)
  OP(*=)
  OP(/=)
#undef OP

// Binary operators overloading
#define OP(type, op)                                                           \
  friend type operator op(const bfloat16 &lhs, const bfloat16 &rhs) {          \
    return type{static_cast<float>(lhs) op static_cast<float>(rhs)};           \
  }                                                                            \
  template <typename T>                                                        \
  friend std::enable_if_t<std::is_convertible_v<T, float>, type> operator op(  \
      const bfloat16 & lhs, const T & rhs) {                                   \
    return type{static_cast<float>(lhs) op static_cast<float>(rhs)};           \
  }                                                                            \
  template <typename T>                                                        \
  friend std::enable_if_t<std::is_convertible_v<T, float>, type> operator op(  \
      const T & lhs, const bfloat16 & rhs) {                                   \
    return type{static_cast<float>(lhs) op static_cast<float>(rhs)};           \
  }
  OP(bfloat16, +)
  OP(bfloat16, -)
  OP(bfloat16, *)
  OP(bfloat16, /)
  OP(bool, ==)
  OP(bool, !=)
  OP(bool, <)
  OP(bool, >)
  OP(bool, <=)
  OP(bool, >=)
#undef OP

  // Bitwise(|,&,~,^), modulo(%) and shift(<<,>>) operations are not supported
  // for floating-point types.

  // Stream Operator << and >>
  inline friend std::ostream &operator<<(std::ostream &O, bfloat16 const &rhs) {
    O << static_cast<float>(rhs);
    return O;
  }

  inline friend std::istream &operator>>(std::istream &I, bfloat16 &rhs) {
    float ValFloat = 0.0f;
    I >> ValFloat;
    rhs = ValFloat;
    return I;
  }
};

namespace detail {

template <int N> void FloatVecToBF16Vec(float src[N], bfloat16 dst[N]) {
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
  uint16_t *dst_i16 = sycl::bit_cast<uint16_t *>(dst);
  if constexpr (N == 1)
    __devicelib_ConvertFToBF16INTELVec1(src, dst_i16);
  else if constexpr (N == 2)
    __devicelib_ConvertFToBF16INTELVec2(src, dst_i16);
  else if constexpr (N == 3)
    __devicelib_ConvertFToBF16INTELVec3(src, dst_i16);
  else if constexpr (N == 4)
    __devicelib_ConvertFToBF16INTELVec4(src, dst_i16);
  else if constexpr (N == 8)
    __devicelib_ConvertFToBF16INTELVec8(src, dst_i16);
  else if constexpr (N == 16)
    __devicelib_ConvertFToBF16INTELVec16(src, dst_i16);
#else
  for (int i = 0; i < N; ++i) {
    // No need to cast as bfloat16 has a assignment op overload that takes
    // a float.
    dst[i] = src[i];
  }
#endif
}

// Helper function for getting the internal representation of a bfloat16.
inline Bfloat16StorageT bfloat16ToBits(const bfloat16 &Value) {
  return Value.value;
}

// Helper function for creating a float16 from a value with the same type as the
// internal representation.
inline bfloat16 bitsToBfloat16(const Bfloat16StorageT Value) {
  bfloat16 res;
  res.value = Value;
  return res;
}

// Class to convert different data types to Bfloat16
// with different rounding modes.
class ConvertToBfloat16 {

  // The automatic rounding mode is RTE.
  enum SYCLRoundingMode { automatic = 0, rte = 1, rtz = 2, rtp = 3, rtn = 4 };

  // Function to get the most significant bit position of a number.
  template <typename Ty> static size_t get_msb_pos(const Ty &x) {
    assert(x != 0);
    size_t idx = 0;
    Ty mask = ((Ty)1 << (sizeof(Ty) * 8 - 1));
    for (idx = 0; idx < (sizeof(Ty) * 8); ++idx) {
      if ((x & mask) == mask)
        break;
      mask >>= 1;
    }

    return (sizeof(Ty) * 8 - 1 - idx);
  }

  // Helper function to get BF16 from float with different rounding modes.
  // Reference:
  // https://github.com/intel/llvm/blob/sycl/libdevice/imf_bf16.hpp#L30
  static bfloat16
  getBFloat16FromFloatWithRoundingMode(const float &f,
                                       SYCLRoundingMode roundingMode) {

    if (roundingMode == SYCLRoundingMode::automatic ||
        roundingMode == SYCLRoundingMode::rte) {
      // Use the default rounding mode.
      return bfloat16{f};
    } else {
      uint32_t u32_val = sycl::bit_cast<uint32_t>(f);
      uint16_t bf16_sign = static_cast<uint16_t>((u32_val >> 31) & 0x1);
      uint16_t bf16_exp = static_cast<uint16_t>((u32_val >> 23) & 0x7FF);
      uint32_t f_mant = u32_val & 0x7F'FFFF;
      uint16_t bf16_mant = static_cast<uint16_t>(f_mant >> 16);
      // +/-infinity and NAN
      if (bf16_exp == 0xFF) {
        if (!f_mant)
          return bitsToBfloat16(bf16_sign ? 0xFF80 : 0x7F80);
        else
          return bitsToBfloat16((bf16_sign << 15) | (bf16_exp << 7) |
                                bf16_mant);
      }

      // +/-0
      if (!bf16_exp && !f_mant) {
        return bitsToBfloat16(bf16_sign ? 0x8000 : 0x0);
      }

      uint16_t mant_discard = static_cast<uint16_t>(f_mant & 0xFFFF);
      switch (roundingMode) {
      case SYCLRoundingMode::rtn:
        if (bf16_sign && mant_discard)
          bf16_mant++;
        break;
      case SYCLRoundingMode::rtz:
        break;
      case SYCLRoundingMode::rtp:
        if (!bf16_sign && mant_discard)
          bf16_mant++;
        break;

      // Should not reach here. Adding these just to suppress the warning.
      case SYCLRoundingMode::automatic:
      case SYCLRoundingMode::rte:
        break;
      }

      // if overflow happens, bf16_exp will be 0xFF and bf16_mant will be 0,
      // infinity will be returned.
      if (bf16_mant == 0x80) {
        bf16_mant = 0;
        bf16_exp++;
      }

      return bitsToBfloat16((bf16_sign << 15) | (bf16_exp << 7) | bf16_mant);
    }
  }

  // Helper function to get BF16 from unsigned integral data types
  // with different rounding modes.
  // Reference:
  // https://github.com/intel/llvm/blob/sycl/libdevice/imf_bf16.hpp#L302
  template <typename T>
  static bfloat16
  getBFloat16FromUIntegralWithRoundingMode(T &u,
                                           SYCLRoundingMode roundingMode) {

    size_t msb_pos = get_msb_pos(u);
    // return half representation for 1
    if (msb_pos == 0)
      return bitsToBfloat16(0x3F80);

    T mant = u & ((static_cast<T>(1) << msb_pos) - 1);
    // Unsigned integral value can be represented by 1.mant * (2^msb_pos),
    // msb_pos is also the bit number of mantissa, 0 < msb_pos < sizeof(Ty) * 8,
    // exponent of bfloat16 precision value range is [-126, 127].

    uint16_t b_exp = msb_pos;
    uint16_t b_mant;

    if (msb_pos <= 7) {
      // No need to round off if we can losslessly fit the input value in
      // mantissa of bfloat16.
      mant <<= (7 - msb_pos);
      b_mant = static_cast<uint16_t>(mant);
    } else {
      b_mant = static_cast<uint16_t>(mant >> (msb_pos - 7));
      T mant_discard = mant & ((static_cast<T>(1) << (msb_pos - 7)) - 1);
      T mid = static_cast<T>(1) << (msb_pos - 8);
      switch (roundingMode) {
      case SYCLRoundingMode::automatic:
      case SYCLRoundingMode::rte:
        if ((mant_discard > mid) ||
            ((mant_discard == mid) && ((b_mant & 0x1) == 0x1)))
          b_mant++;
        break;
      case SYCLRoundingMode::rtp:
        if (mant_discard)
          b_mant++;
        break;
      case SYCLRoundingMode::rtn:
      case SYCLRoundingMode::rtz:
        break;
      }
    }
    if (b_mant == 0x80) {
      b_exp++;
      b_mant = 0;
    }

    b_exp += 127;
    return bitsToBfloat16((b_exp << 7) | b_mant);
  }

  // Helper function to get BF16 from signed integral data types.
  // Reference:
  // https://github.com/intel/llvm/blob/sycl/libdevice/imf_bf16.hpp#L353
  template <typename T>
  static bfloat16
  getBFloat16FromSIntegralWithRoundingMode(T &i,
                                           SYCLRoundingMode roundingMode) {
    // Get unsigned type corresponding to T.
    typedef typename std::make_unsigned_t<T> UTy;

    uint16_t b_sign = (i >= 0) ? 0 : 0x8000;
    UTy ui = (i > 0) ? static_cast<UTy>(i) : static_cast<UTy>(-i);
    size_t msb_pos = get_msb_pos<UTy>(ui);
    if (msb_pos == 0)
      return bitsToBfloat16(b_sign ? 0xBF80 : 0x3F80);
    UTy mant = ui & ((static_cast<UTy>(1) << msb_pos) - 1);

    uint16_t b_exp = msb_pos;
    uint16_t b_mant;
    if (msb_pos <= 7) {
      mant <<= (7 - msb_pos);
      b_mant = static_cast<uint16_t>(mant);
    } else {
      b_mant = static_cast<uint16_t>(mant >> (msb_pos - 7));
      T mant_discard = mant & ((static_cast<T>(1) << (msb_pos - 7)) - 1);
      T mid = static_cast<T>(1) << (msb_pos - 8);
      switch (roundingMode) {
      case SYCLRoundingMode::automatic:
      case SYCLRoundingMode::rte:
        if ((mant_discard > mid) ||
            ((mant_discard == mid) && ((b_mant & 0x1) == 0x1)))
          b_mant++;
        break;
      case SYCLRoundingMode::rtp:
        if (mant_discard && !b_sign)
          b_mant++;
        break;
      case SYCLRoundingMode::rtn:
        if (mant_discard && b_sign)
          b_mant++;
      case SYCLRoundingMode::rtz:
        break;
      }
    }

    if (b_mant == 0x80) {
      b_exp++;
      b_mant = 0;
    }
    b_exp += 127;
    return bitsToBfloat16(b_sign | (b_exp << 7) | b_mant);
  }

  // Helper function to get BF16 from double with RTE rounding modes.
  // Reference:
  // https://github.com/intel/llvm/blob/sycl/libdevice/imf_bf16.hpp#L79
  static bfloat16 getBFloat16FromDoubleWithRTE(const double &d) {

    uint64_t u64_val = sycl::bit_cast<uint64_t>(d);
    int16_t bf16_sign = (u64_val >> 63) & 0x1;
    uint16_t fp64_exp = static_cast<uint16_t>((u64_val >> 52) & 0x7FF);
    uint64_t fp64_mant = (u64_val & 0xF'FFFF'FFFF'FFFF);
    uint16_t bf16_mant;
    // handling +/-infinity and NAN for double input
    if (fp64_exp == 0x7FF) {
      if (!fp64_mant) {
        return bf16_sign ? 0xFF80 : 0x7F80;
      } else {
        // returns a quiet NaN
        return 0x7FC0;
      }
    }

    // Subnormal double precision is converted to 0
    if (fp64_exp == 0) {
      return bf16_sign ? 0x8000 : 0x0;
    }

    fp64_exp -= 1023;
    // handling overflow, convert to +/-infinity
    if (static_cast<int16_t>(fp64_exp) > 127) {
      return bf16_sign ? 0xFF80 : 0x7F80;
    }

    // handling underflow
    if (static_cast<int16_t>(fp64_exp) < -133) {
      return bf16_sign ? 0x8000 : 0x0;
    }

    //-133 <= fp64_exp <= 127, 1.signicand * 2^fp64_exp
    // For these numbers, they are NOT subnormal double-precision numbers but
    // will turn into subnormal when converting to bfloat16
    uint64_t discard_bits;
    if (static_cast<int16_t>(fp64_exp) < -126) {
      fp64_mant |= 0x10'0000'0000'0000;
      fp64_mant >>= -126 - static_cast<int16_t>(fp64_exp) - 1;
      discard_bits = fp64_mant & 0x3FFF'FFFF'FFFF;
      bf16_mant = static_cast<uint16_t>(fp64_mant >> 46);
      if (discard_bits > 0x2000'0000'0000 ||
          ((discard_bits == 0x2000'0000'0000) && ((bf16_mant & 0x1) == 0x1)))
        bf16_mant += 1;
      fp64_exp = 0;
      if (bf16_mant == 0x80) {
        bf16_mant = 0;
        fp64_exp = 1;
      }
      return (bf16_sign << 15) | (fp64_exp << 7) | bf16_mant;
    }

    // For normal value, discard 45 bits from mantissa
    discard_bits = fp64_mant & 0x1FFF'FFFF'FFFF;
    bf16_mant = static_cast<uint16_t>(fp64_mant >> 45);
    if (discard_bits > 0x1000'0000'0000 ||
        ((discard_bits == 0x1000'0000'0000) && ((bf16_mant & 0x1) == 0x1)))
      bf16_mant += 1;

    if (bf16_mant == 0x80) {
      if (fp64_exp != 127) {
        bf16_mant = 0;
        fp64_exp++;
      } else {
        return bf16_sign ? 0xFF80 : 0x7F80;
      }
    }
    fp64_exp += 127;

    return (bf16_sign << 15) | (fp64_exp << 7) | bf16_mant;
  }

public:
  template <typename Ty, int rm>
  static bfloat16 getBfloat16WithRoundingMode(const Ty &a) {

    if (!a)
      return bfloat16{0.0f};

    constexpr SYCLRoundingMode roundingMode = static_cast<SYCLRoundingMode>(rm);

    // Float.
    if constexpr (std::is_same_v<Ty, float>) {
      return getBFloat16FromFloatWithRoundingMode(a, roundingMode);
    }
    // Double.
    else if constexpr (std::is_same_v<Ty, double>) {
      static_assert(
          roundingMode == SYCLRoundingMode::automatic ||
              roundingMode == SYCLRoundingMode::rte,
          "Only automatic/RTE rounding mode is supported for double type.");
      return getBFloat16FromDoubleWithRTE(a);
    }
    // Half
    else if constexpr (std::is_same_v<Ty, sycl::half>) {
      // Convert half to float and then convert to bfloat16.
      // Conversion of half to float is lossless as the latter
      // have a wider dynamic range.
      return getBFloat16FromFloatWithRoundingMode(static_cast<float>(a),
                                                  roundingMode);
    }
    // Unsigned integral types.
    else if constexpr (std::is_integral_v<Ty> && std::is_unsigned_v<Ty>) {
      return getBFloat16FromUIntegralWithRoundingMode<Ty>(a, roundingMode);
    }
    // Signed integral types.
    else if constexpr (std::is_integral_v<Ty> && std::is_signed_v<Ty>) {
      return getBFloat16FromSIntegralWithRoundingMode<Ty>(a, roundingMode);
    } else {
      static_assert(std::is_integral_v<Ty> || std::is_floating_point_v<Ty>,
                    "Only integral and floating point types are supported.");
    }
  }
}; // class ConvertToBfloat16.
} // namespace detail

} // namespace ext::oneapi

} // namespace _V1
} // namespace sycl
