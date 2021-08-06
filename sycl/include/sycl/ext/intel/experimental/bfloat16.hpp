//==--------- bfloat16.hpp ------- SYCL bfloat16 conversion ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {

class [[sycl_detail::uses_aspects(ext_intel_bf16_conversion)]] bfloat16 {
  using storage_t = uint16_t;
  storage_t value;

public:
  bfloat16() = default;
  bfloat16(const bfloat16 &) = default;
  ~bfloat16() = default;

  // Explicit conversion functions
  static storage_t from_float(const float &a) {
#if defined(__SYCL_DEVICE_ONLY__)
    return __spirv_ConvertFToBF16INTEL(a);
#else
    throw runtime_error("Bfloat16 conversion is not supported on HOST device.",
                        PI_INVALID_DEVICE);
#endif
  }
  static float to_float(const storage_t &a) {
#if defined(__SYCL_DEVICE_ONLY__)
    return __spirv_ConvertBF16ToFINTEL(a);
#else
    throw runtime_error("Bfloat16 conversion is not supported on HOST device.",
                        PI_INVALID_DEVICE);
#endif
  }

  // Direct initialization
  bfloat16(const storage_t &a) : value(a) {}

  // Implicit conversion from float to bfloat16
  bfloat16(const float &a) { value = from_float(a); }

  bfloat16 &operator=(const float &rhs) {
    value = from_float(rhs);
    return *this;
  }

  // Implicit conversion from bfloat16 to float
  operator float() const { return to_float(value); }

  // Get raw bits representation of bfloat16
  operator storage_t() const { return value; }

// Assignment operators overloading
#define OP(op)                                                                 \
  friend bfloat16 operator op(bfloat16 &lhs, const bfloat16 &rhs) {            \
    float f = static_cast<float>(lhs);                                         \
    f op static_cast<float>(rhs);                                              \
    return lhs = f;                                                            \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  friend bfloat16 operator op(bfloat16 &lhs, const T &rhs) {                   \
    float f = static_cast<float>(lhs);                                         \
                                                                               \
    f op static_cast<float>(rhs);                                              \
    return lhs = f;                                                            \
  }                                                                            \
  template <typename T> friend T operator op(T &lhs, const bfloat16 &rhs) {    \
    float f = static_cast<float>(lhs);                                         \
    f op static_cast<float>(rhs);                                              \
    return lhs = f;                                                            \
  }
  OP(+=)
  OP(-=)
  OP(*=)
  OP(/=)
#undef OP

// Increment and decrement operators overloading
#define OP(op)                                                                 \
  bfloat16 &operator op() {                                                    \
    float f = to_float(value);                                                 \
    value = from_float(op f);                                                  \
    return *this;                                                              \
  }                                                                            \
  bfloat16 operator op(int) {                                                  \
    bfloat16 old = *this;                                                      \
    operator op();                                                             \
    return old;                                                                \
  }
  OP(++)
  OP(--)
#undef OP

  // Unary minus operator overloading
  bfloat16 operator-() { return bfloat16{-to_float(value)}; }

  // Logical operators (!,||,&&) are covered if we can cast to bool
  explicit operator bool() { return to_float(value) != 0.0f; }

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

} // namespace experimental
} // namespace intel
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::intel' instead") INTEL {
  using namespace ext::intel;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
