//==-------------- half_type.hpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/bit_cast.hpp>              // for bit_cast
#include <sycl/detail/export.hpp>         // for __SYCL_EXPORT
#include <sycl/detail/iostream_proxy.hpp> // for istream, ostream
#include <sycl/detail/vector_traits.hpp>  // for vector_alignment

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/aspects.hpp>
#endif

#include <cstddef>     // for size_t
#include <cstdint>     // for uint16_t, uint32_t, uint8_t
#include <functional>  // for hash
#include <limits>      // for float_denorm_style, float_r...
#include <string_view> // for hash
#include <type_traits> // for enable_if_t

#if !defined(__has_builtin) || !__has_builtin(__builtin_expect)
#define __builtin_expect(a, b) (a)
#endif

#ifdef __SYCL_DEVICE_ONLY__
// `constexpr` could work because the implicit conversion from `float` to
// `_Float16` can be `constexpr`.
#define __SYCL_CONSTEXPR_HALF constexpr
#elif __cpp_lib_bit_cast ||                                                    \
    (defined(__has_builtin) && __has_builtin(__builtin_bit_cast))
#define __SYCL_CONSTEXPR_HALF constexpr
#else
#define __SYCL_CONSTEXPR_HALF
#endif

namespace sycl {
inline namespace _V1 {
namespace detail::half_impl {
class half;
}
using half = detail::half_impl::half;

namespace ext::intel::esimd::detail {
class WrapperElementTypeProxy;
} // namespace ext::intel::esimd::detail

namespace detail {

inline __SYCL_CONSTEXPR_HALF uint16_t float2Half(const float &Val) {
  const uint32_t Bits = sycl::bit_cast<uint32_t>(Val);

  // Extract the sign from the float value
  const uint16_t Sign = (Bits & 0x80000000) >> 16;
  // Extract the fraction from the float value
  const uint32_t Frac32 = Bits & 0x7fffff;
  // Extract the exponent from the float value
  const uint8_t Exp32 = (Bits & 0x7f800000) >> 23;
  const int16_t Exp32Diff = Exp32 - 127;

  // initialize to 0, covers the case for 0 and small numbers
  uint16_t Exp16 = 0, Frac16 = 0;

  if (__builtin_expect(Exp32Diff > 15, 0)) {
    // Infinity and big numbers convert to infinity
    Exp16 = 0x1f;
  } else if (__builtin_expect(Exp32Diff > -14, 0)) {
    // normal range for half type
    Exp16 = Exp32Diff + 15;
    // convert 23-bit mantissa to 10-bit mantissa.
    Frac16 = Frac32 >> 13;
    // Round the mantissa as given in OpenCL spec section : 6.1.1.1 The half
    // data type.
    // Round to nearest.
    uint32_t roundBits = Frac32 & 0x1fff;
    uint32_t halfway = 0x1000;
    if (roundBits > halfway)
      Frac16 += 1;
    // Tie to even.
    else if (roundBits == halfway)
      Frac16 += Frac16 & 1;
  } else if (__builtin_expect(Exp32Diff > -25, 0)) {
    // subnormals
    Frac16 = (Frac32 | (uint32_t(1) << 23)) >> (-Exp32Diff - 1);
  }

  if (__builtin_expect(Exp32 == 0xff && Frac32 != 0, 0)) {
    // corner case: FP32 is NaN
    Exp16 = 0x1F;
    Frac16 = 0x200;
  }

  // Compose the final FP16 binary
  uint16_t Ret = 0;
  Ret |= Sign;
  Ret |= Exp16 << 10;
  Ret += Frac16; // Add the carry bit from operation Frac16 += 1;

  return Ret;
}

inline __SYCL_CONSTEXPR_HALF float half2Float(const uint16_t &Val) {
  // Extract the sign from the bits. It is 1 if the sign is negative
  const uint32_t Sign = static_cast<uint32_t>(Val & 0x8000) << 16;
  // Extract the exponent from the bits
  const uint8_t Exp16 = (Val & 0x7c00) >> 10;
  // Extract the fraction from the bits
  uint16_t Frac16 = Val & 0x3ff;

  uint32_t Exp32 = 0;
  if (__builtin_expect(Exp16 == 0x1f, 0)) {
    Exp32 = 0xff;
  } else if (__builtin_expect(Exp16 == 0, 0)) {
    Exp32 = 0;
  } else {
    Exp32 = static_cast<uint32_t>(Exp16) + 112;
  }
  // corner case: subnormal -> normal
  // The denormal number of FP16 can be represented by FP32, therefore we need
  // to recover the exponent and recalculate the fration.
  if (__builtin_expect(Exp16 == 0 && Frac16 != 0, 0)) {
    uint8_t OffSet = 0;
    do {
      ++OffSet;
      Frac16 <<= 1;
    } while ((Frac16 & 0x400) != 0x400);
    // mask the 9th bit
    Frac16 &= 0x3ff;
    Exp32 = 113 - OffSet;
  }

  uint32_t Frac32 = Frac16 << 13;

  uint32_t Bits = 0;
  Bits |= Sign;
  Bits |= (Exp32 << 23);
  Bits |= Frac32;
  const float Result = sycl::bit_cast<float>(Bits);
  return Result;
}

namespace half_impl {
class half;

// Several aliases are defined below:
// - StorageT: actual representation of half data type. It is used by scalar
//   half values. On device side, it points to some native half data type, while
//   on host it is represented by a 16-bit integer that the implementation
//   manipulates to emulate half-precision floating-point behavior.
//
// - BIsRepresentationT: data type which is used by built-in functions. It is
//   distinguished from StorageT, because on host, we can still operate on the
//   wrapper itself and there is no sense in direct usage of underlying data
//   type (too many changes required for BIs implementation without any
//   foreseeable profits)
//
// - VecElemT: representation of each element in the vector. On device it is
//   the same as StorageT to carry a native vector representation, while on
//   host it stores the sycl::half implementation directly.
//
// - VecNStorageT: representation of N-element vector of halfs. Follows the
//   same logic as VecElemT.
#ifdef __SYCL_DEVICE_ONLY__
using StorageT = _Float16;
using BIsRepresentationT = _Float16;
using VecElemT = _Float16;
#else // SYCL_DEVICE_ONLY
using StorageT = uint16_t;
// No need to extract underlying data type for built-in functions operating on
// host
using BIsRepresentationT = half;
using VecElemT = half;
#endif // SYCL_DEVICE_ONLY

// Creation token to disambiguate constructors.
struct RawHostHalfToken {
  constexpr explicit RawHostHalfToken(uint16_t Val) : Value{Val} {}
  uint16_t Value;
};

#ifndef __SYCL_DEVICE_ONLY__
class half {
#else
class [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] half {
#endif
public:
  half() = default;
  constexpr half(const half &) = default;
  constexpr half(half &&) = default;

#ifdef __SYCL_DEVICE_ONLY__
  __SYCL_CONSTEXPR_HALF half(const float &rhs) : Data(rhs) {}
#else
  __SYCL_CONSTEXPR_HALF half(const float &rhs) : Data(float2Half(rhs)) {}
#endif // __SYCL_DEVICE_ONLY__

  constexpr half &operator=(const half &rhs) = default;

  // Operator +=, -=, *=, /=
#ifdef __SYCL_DEVICE_ONLY__
  __SYCL_CONSTEXPR_HALF half &operator+=(const half &rhs) {
    Data += rhs.Data;
    return *this;
  }

  __SYCL_CONSTEXPR_HALF half &operator-=(const half &rhs) {
    Data -= rhs.Data;
    return *this;
  }

  __SYCL_CONSTEXPR_HALF half &operator*=(const half &rhs) {
    Data *= rhs.Data;
    return *this;
  }

  __SYCL_CONSTEXPR_HALF half &operator/=(const half &rhs) {
    Data /= rhs.Data;
    return *this;
  }
#else
  __SYCL_CONSTEXPR_HALF half &operator+=(const half &rhs) {
    *this = operator float() + static_cast<float>(rhs);
    return *this;
  }

  __SYCL_CONSTEXPR_HALF half &operator-=(const half &rhs) {
    *this = operator float() - static_cast<float>(rhs);
    return *this;
  }

  __SYCL_CONSTEXPR_HALF half &operator*=(const half &rhs) {
    *this = operator float() * static_cast<float>(rhs);
    return *this;
  }

  __SYCL_CONSTEXPR_HALF half &operator/=(const half &rhs) {
    *this = operator float() / static_cast<float>(rhs);
    return *this;
  }
#endif // __SYCL_DEVICE_ONLY__

  // Operator ++, --
  __SYCL_CONSTEXPR_HALF half &operator++() {
    *this += 1;
    return *this;
  }

  __SYCL_CONSTEXPR_HALF half operator++(int) {
    half ret(*this);
    operator++();
    return ret;
  }

  __SYCL_CONSTEXPR_HALF half &operator--() {
    *this -= 1;
    return *this;
  }

  __SYCL_CONSTEXPR_HALF half operator--(int) {
    half ret(*this);
    operator--();
    return ret;
  }

  // Operator neg
#ifdef __SYCL_DEVICE_ONLY__
  __SYCL_CONSTEXPR_HALF friend half operator-(const half other) {
    return half(-other.Data);
  }
#else
  __SYCL_CONSTEXPR_HALF friend half operator-(const half other) {
    return half(RawHostHalfToken(other.Data ^ 0x8000));
  }
#endif // __SYCL_DEVICE_ONLY__

// Operator +, -, *, /
#define OP(op, op_eq)                                                          \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half lhs,                \
                                                const half rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend double operator op(const half lhs,              \
                                                  const double rhs) {          \
    double rtn = lhs;                                                          \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend double operator op(const double lhs,            \
                                                  const half rhs) {            \
    double rtn = lhs;                                                          \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend float operator op(const half lhs,               \
                                                 const float rhs) {            \
    float rtn = lhs;                                                           \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend float operator op(const float lhs,              \
                                                 const half rhs) {             \
    float rtn = lhs;                                                           \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half lhs,                \
                                                const int rhs) {               \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const int lhs,                 \
                                                const half rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half lhs,                \
                                                const long rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const long lhs,                \
                                                const half rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half lhs,                \
                                                const long long rhs) {         \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const long long lhs,           \
                                                const half rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half &lhs,               \
                                                const unsigned int &rhs) {     \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const unsigned int &lhs,       \
                                                const half &rhs) {             \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half &lhs,               \
                                                const unsigned long &rhs) {    \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const unsigned long &lhs,      \
                                                const half &rhs) {             \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(                               \
      const half &lhs, const unsigned long long &rhs) {                        \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const unsigned long long &lhs, \
                                                const half &rhs) {             \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }
  OP(+, +=)
  OP(-, -=)
  OP(*, *=)
  OP(/, /=)

#undef OP

// Operator ==, !=, <, >, <=, >=
#define OP(op)                                                                 \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const half &lhs,               \
                                                const half &rhs) {             \
    return lhs.getFPRep() op rhs.getFPRep();                                   \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const half &lhs,               \
                                                const double &rhs) {           \
    return lhs.getFPRep() op rhs;                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const double &lhs,             \
                                                const half &rhs) {             \
    return lhs op rhs.getFPRep();                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const half &lhs,               \
                                                const float &rhs) {            \
    return lhs.getFPRep() op rhs;                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const float &lhs,              \
                                                const half &rhs) {             \
    return lhs op rhs.getFPRep();                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const half &lhs,               \
                                                const int &rhs) {              \
    return lhs.getFPRep() op rhs;                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const int &lhs,                \
                                                const half &rhs) {             \
    return lhs op rhs.getFPRep();                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const half &lhs,               \
                                                const long &rhs) {             \
    return lhs.getFPRep() op rhs;                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const long &lhs,               \
                                                const half &rhs) {             \
    return lhs op rhs.getFPRep();                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const half &lhs,               \
                                                const long long &rhs) {        \
    return lhs.getFPRep() op rhs;                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const long long &lhs,          \
                                                const half &rhs) {             \
    return lhs op rhs.getFPRep();                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const half &lhs,               \
                                                const unsigned int &rhs) {     \
    return lhs.getFPRep() op rhs;                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const unsigned int &lhs,       \
                                                const half &rhs) {             \
    return lhs op rhs.getFPRep();                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const half &lhs,               \
                                                const unsigned long &rhs) {    \
    return lhs.getFPRep() op rhs;                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const unsigned long &lhs,      \
                                                const half &rhs) {             \
    return lhs op rhs.getFPRep();                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(                               \
      const half &lhs, const unsigned long long &rhs) {                        \
    return lhs.getFPRep() op rhs;                                              \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend bool operator op(const unsigned long long &lhs, \
                                                const half &rhs) {             \
    return lhs op rhs.getFPRep();                                              \
  }
  OP(==)
  OP(!=)
  OP(<)
  OP(>)
  OP(<=)
  OP(>=)

#undef OP

  // Operator float
#ifdef __SYCL_DEVICE_ONLY__
  __SYCL_CONSTEXPR_HALF operator float() const {
    return static_cast<float>(Data);
  }
#else
  __SYCL_CONSTEXPR_HALF operator float() const { return half2Float(Data); }
#endif // __SYCL_DEVICE_ONLY__

  // Operator << and >>
  inline friend std::ostream &operator<<(std::ostream &O,
                                         sycl::half const &rhs) {
    O << static_cast<float>(rhs);
    return O;
  }

  inline friend std::istream &operator>>(std::istream &I, sycl::half &rhs) {
    float ValFloat = 0.0f;
    I >> ValFloat;
    rhs = ValFloat;
    return I;
  }

  template <typename Key> friend struct std::hash;

  friend class sycl::ext::intel::esimd::detail::WrapperElementTypeProxy;

private:
  // When doing operations, we cannot simply work with Data on host as
  // it is an integer. Instead, convert it to float. On device we can work with
  // Data as it is already a floating point representation.
#ifdef __SYCL_DEVICE_ONLY__
  __SYCL_CONSTEXPR_HALF StorageT getFPRep() const { return Data; }
#else
  __SYCL_CONSTEXPR_HALF float getFPRep() const { return operator float(); }
#endif

#ifndef __SYCL_DEVICE_ONLY__
  // Because sycl::bit_cast might not be constexpr on certain systems,
  // implementation needs shortcut for creating a host sycl::half directly from
  // a uint16_t representation.
  constexpr explicit half(RawHostHalfToken X) : Data(X.Value) {}

  friend constexpr inline half CreateHostHalfRaw(uint16_t X);
#endif // __SYCL_DEVICE_ONLY__

  StorageT Data;
};

#ifndef __SYCL_DEVICE_ONLY__
constexpr inline half CreateHostHalfRaw(uint16_t X) {
  return half(RawHostHalfToken(X));
}
#endif // __SYCL_DEVICE_ONLY__
} // namespace half_impl

// According to the C++ standard, math functions from cmath/math.h should work
// only on arithmetic types. We can't specify half type as arithmetic/floating
// point(via std::is_floating_point) since only float, double and long double
// types are "floating point" according to the standard. In order to use half
// type with these math functions we cast half to float using template
// function helper.
template <typename T> inline T cast_if_host_half(T val) { return val; }

inline float cast_if_host_half(half_impl::half val) {
  return static_cast<float>(val);
}

} // namespace detail

} // namespace _V1
} // namespace sycl

// Partial specialization of some functions in namespace `std`
namespace std {

// Partial specialization of `std::hash<sycl::half>`
template <> struct hash<sycl::half> {
  size_t operator()(sycl::half const &Key) const noexcept {
    return hash<uint16_t>{}(reinterpret_cast<const uint16_t &>(Key));
  }
};

// Partial specialization of `std::numeric<sycl::half>`
template <> struct numeric_limits<sycl::half> {
  // All following values are either calculated based on description of each
  // function/value on https://en.cppreference.com/w/cpp/types/numeric_limits,
  // or cl_platform.h.
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr bool tinyness_before = false;
  static constexpr bool traps = false;
  static constexpr int max_exponent10 = 4;
  static constexpr int max_exponent = 16;
  static constexpr int min_exponent10 = -4;
  static constexpr int min_exponent = -13;
  static constexpr int radix = 2;
  static constexpr int max_digits10 = 5;
  static constexpr int digits = 11;
  static constexpr bool is_bounded = true;
  static constexpr int digits10 = 3;
  static constexpr bool is_modulo = false;
  static constexpr bool is_iec559 = true;
  static constexpr float_round_style round_style = round_to_nearest;

  static __SYCL_CONSTEXPR_HALF const sycl::half(min)() noexcept {
    return 6.103515625e-05f; // half minimum value
  }

  static __SYCL_CONSTEXPR_HALF const sycl::half(max)() noexcept {
    return 65504.0f; // half maximum value
  }

  static __SYCL_CONSTEXPR_HALF const sycl::half lowest() noexcept {
    return -65504.0f; // -1*(half maximum value)
  }

  static __SYCL_CONSTEXPR_HALF const sycl::half epsilon() noexcept {
    return 9.765625e-04f; // half epsilon
  }

  static __SYCL_CONSTEXPR_HALF const sycl::half round_error() noexcept {
    return 0.5f;
  }

  static constexpr const sycl::half infinity() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_huge_valf();
#else
    return sycl::detail::half_impl::CreateHostHalfRaw(
        static_cast<uint16_t>(0x7C00));
#endif
  }

  static __SYCL_CONSTEXPR_HALF const sycl::half quiet_NaN() noexcept {
    return __builtin_nanf("");
  }

  static __SYCL_CONSTEXPR_HALF const sycl::half signaling_NaN() noexcept {
    return __builtin_nansf("");
  }

  static __SYCL_CONSTEXPR_HALF const sycl::half denorm_min() noexcept {
    return 5.96046e-08f;
  }
};

} // namespace std

#undef __SYCL_CONSTEXPR_HALF
#undef _CPP14_CONSTEXPR
