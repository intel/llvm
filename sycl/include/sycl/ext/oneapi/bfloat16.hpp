//==--------- bfloat16.hpp ------- SYCL bfloat16 conversion ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/half_type.hpp>

#if !defined(__SYCL_DEVICE_ONLY__)
#include <cmath>
#endif

extern "C" SYCL_EXTERNAL uint16_t
__devicelib_ConvertFToBF16INTEL(const float &) noexcept;
extern "C" SYCL_EXTERNAL float
__devicelib_ConvertBF16ToFINTEL(const uint16_t &) noexcept;

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
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
  // Explicit conversion functions
  static detail::Bfloat16StorageT from_float(const float &a) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
#if (__CUDA_ARCH__ >= 800)
    return __nvvm_f2bf16_rn(a);
#else
    // TODO find a better way to check for NaN
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
#endif
#else
    return __devicelib_ConvertFToBF16INTEL(a);
#endif
#else
    // In case float value is nan - propagate bfloat16's qnan
    if (std::isnan(a))
      return 0xffc1;
    union {
      uint32_t intStorage;
      float floatValue;
    };
    floatValue = a;
    // Do RNE and truncate
    uint32_t roundingBias = ((intStorage >> 16) & 0x1) + 0x00007FFF;
    return static_cast<uint16_t>((intStorage + roundingBias) >> 16);
#endif
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
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
#if (__CUDA_ARCH__ >= 800)
    return detail::bitsToBfloat16(__nvvm_neg_bf16(lhs.value));
#else
    return -to_float(lhs.value);
#endif
#else
    return bfloat16{-__devicelib_ConvertBF16ToFINTEL(lhs.value)};
#endif
#else
    return -to_float(lhs.value);
#endif
  }

// Increment and decrement operators overloading
#define OP(op)                                                                 \
  friend bfloat16 &operator op(bfloat16 &lhs) {                                \
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
  friend bfloat16 &operator op(bfloat16 &lhs, const bfloat16 &rhs) {           \
    float f = static_cast<float>(lhs);                                         \
    f op static_cast<float>(rhs);                                              \
    return lhs = f;                                                            \
  }                                                                            \
  template <typename T>                                                        \
  friend bfloat16 &operator op(bfloat16 &lhs, const T &rhs) {                  \
    float f = static_cast<float>(lhs);                                         \
    f op static_cast<float>(rhs);                                              \
    return lhs = f;                                                            \
  }                                                                            \
  template <typename T> friend T &operator op(T &lhs, const bfloat16 &rhs) {   \
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
  friend type operator op(const bfloat16 &lhs, const T &rhs) {                 \
    return type{static_cast<float>(lhs) op static_cast<float>(rhs)};           \
  }                                                                            \
  template <typename T>                                                        \
  friend type operator op(const T &lhs, const bfloat16 &rhs) {                 \
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

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
