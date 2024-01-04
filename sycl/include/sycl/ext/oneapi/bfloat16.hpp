//==--------- bfloat16.hpp ------- SYCL bfloat16 conversion ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aliases.hpp>                   // for half
#include <sycl/builtins.hpp>                  // for isnan
#include <sycl/detail/defines_elementary.hpp> // for __DPCPP_SYCL_EXTERNAL
#include <sycl/half_type.hpp>                 // for half

#include <stdint.h> // for uint16_t, uint32_t

extern "C" __DPCPP_SYCL_EXTERNAL uint16_t
__devicelib_ConvertFToBF16INTEL(const float &) noexcept;
extern "C" __DPCPP_SYCL_EXTERNAL float
__devicelib_ConvertBF16ToFINTEL(const uint16_t &) noexcept;

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {

class bfloat16;

namespace detail {
using Bfloat16StorageT = uint16_t;
Bfloat16StorageT bfloat16ToBits(const bfloat16 &Value);
bfloat16 bitsToBfloat16(const Bfloat16StorageT Value);
} // namespace detail

class bfloat16 {
  detail::Bfloat16StorageT value;

  friend inline detail::Bfloat16StorageT
  detail::bfloat16ToBits(const bfloat16 &Value);
  friend inline bfloat16
  detail::bitsToBfloat16(const detail::Bfloat16StorageT Value);

public:
  bfloat16() = default;
  bfloat16(const bfloat16 &) = default;
  ~bfloat16() = default;

private:
  static detail::Bfloat16StorageT from_float_fallback(const float &a) {
    if (sycl::isnan(a))
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
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
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
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    return bfloat16{-__devicelib_ConvertBF16ToFINTEL(lhs.value)};
#else
    return -to_float(lhs.value);
#endif
  }

  bfloat16 &operator+=(const bfloat16 &rhs) {
    value = from_float(to_float(value) + to_float(rhs.value));
    return *this;
  }

  bfloat16 &operator-=(const bfloat16 &rhs) {
    value = from_float(to_float(value) - to_float(rhs.value));
    return *this;
  }

  bfloat16 &operator*=(const bfloat16 &rhs) {
    value = from_float(to_float(value) * to_float(rhs.value));
    return *this;
  }

  bfloat16 &operator/=(const bfloat16 &rhs) {
    value = from_float(to_float(value) / to_float(rhs.value));
    return *this;
  }

  // Operator ++, --
  bfloat16 &operator++() {
    float f = to_float(value);
    value = from_float(++f);
    return *this;
  }

  bfloat16 operator++(int) {
    bfloat16 ret(*this);
    operator++();
    return ret;
  }

  bfloat16 &operator--() {
    float f = to_float(value);
    value = from_float(--f);
    return *this;
  }

  bfloat16 operator--(int) {
    bfloat16 ret(*this);
    operator--();
    return ret;
  }

// Operator +, -, *, /
#define OP(op)                                                                 \
  friend bfloat16 operator op(const bfloat16 lhs, const bfloat16 rhs) {        \
    return to_float(lhs.value) op to_float(rhs.value);                         \
  }                                                                            \
  friend double operator op(const bfloat16 lhs, const double rhs) {            \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend double operator op(const double lhs, const bfloat16 rhs) {            \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend float operator op(const bfloat16 lhs, const float rhs) {              \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend float operator op(const float lhs, const bfloat16 rhs) {              \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bfloat16 operator op(const bfloat16 lhs, const int rhs) {             \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bfloat16 operator op(const int lhs, const bfloat16 rhs) {             \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bfloat16 operator op(const bfloat16 lhs, const long rhs) {            \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bfloat16 operator op(const long lhs, const bfloat16 rhs) {            \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bfloat16 operator op(const bfloat16 lhs, const long long rhs) {       \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bfloat16 operator op(const long long lhs, const bfloat16 rhs) {       \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bfloat16 operator op(const bfloat16 &lhs, const unsigned int &rhs) {  \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bfloat16 operator op(const unsigned int &lhs, const bfloat16 &rhs) {  \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bfloat16 operator op(const bfloat16 &lhs, const unsigned long &rhs) { \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bfloat16 operator op(const unsigned long &lhs, const bfloat16 &rhs) { \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bfloat16 operator op(const bfloat16 &lhs,                             \
                              const unsigned long long &rhs) {                 \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bfloat16 operator op(const unsigned long long &lhs,                   \
                              const bfloat16 &rhs) {                           \
    return lhs op to_float(rhs.value);                                         \
  }
  OP(+)
  OP(-)
  OP(*)
  OP(/)

#undef OP

// Operator ==, !=, <, >, <=, >=
#define OP(op)                                                                 \
  friend bool operator op(const bfloat16 &lhs, const bfloat16 &rhs) {          \
    return to_float(lhs.value) op to_float(rhs.value);                         \
  }                                                                            \
  friend bool operator op(const bfloat16 &lhs, const double &rhs) {            \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bool operator op(const double &lhs, const bfloat16 &rhs) {            \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bool operator op(const bfloat16 &lhs, const float &rhs) {             \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bool operator op(const float &lhs, const bfloat16 &rhs) {             \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bool operator op(const bfloat16 &lhs, const int &rhs) {               \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bool operator op(const int &lhs, const bfloat16 &rhs) {               \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bool operator op(const bfloat16 &lhs, const long &rhs) {              \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bool operator op(const long &lhs, const bfloat16 &rhs) {              \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bool operator op(const bfloat16 &lhs, const long long &rhs) {         \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bool operator op(const long long &lhs, const bfloat16 &rhs) {         \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bool operator op(const bfloat16 &lhs, const unsigned int &rhs) {      \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bool operator op(const unsigned int &lhs, const bfloat16 &rhs) {      \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bool operator op(const bfloat16 &lhs, const unsigned long &rhs) {     \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bool operator op(const unsigned long &lhs, const bfloat16 &rhs) {     \
    return lhs op to_float(rhs.value);                                         \
  }                                                                            \
  friend bool operator op(const bfloat16 &lhs,                                 \
                          const unsigned long long &rhs) {                     \
    return to_float(lhs.value) op rhs;                                         \
  }                                                                            \
  friend bool operator op(const unsigned long long &lhs,                       \
                          const bfloat16 &rhs) {                               \
    return lhs op to_float(rhs.value);                                         \
  }
  OP(==)
  OP(!=)
  OP(<)
  OP(>)
  OP(<=)
  OP(>=)

#undef OP

  // Bitwise(|,&,~,^), modulo(%) and shift(<<,>>) operations are not supported
  // for floating-point types.
};

namespace detail {

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

} // namespace detail

} // namespace ext::oneapi

} // namespace _V1
} // namespace sycl
