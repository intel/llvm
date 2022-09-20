//==--------- tfloat32.hpp ------- SYCL tensorfloat32 conversion ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/bit_cast.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

class tfloat32 {
  using storage_t = uint32_t;
  storage_t value;

public:
  tfloat32() = default;
  tfloat32(const tfloat32 &) = default;
  ~tfloat32() = default;

  // Explicit conversion functions
  static storage_t from_float(const float &a) {

    storage_t tmp_uint = sycl::bit_cast<storage_t>(a);
    tmp_uint &= 0xFFFFE000u;

    return tmp_uint;
  }
  static float to_float(const storage_t &a) {
    return sycl::bit_cast<float>(a & 0xFFFFE000u);
  }

  // Implicit conversion from float to tfloat32
  tfloat32(const float &a) { value = from_float(a); }

  tfloat32 &operator=(const float &rhs) {
    value = from_float(rhs);
    return *this;
  }

  // Implicit conversion from tfloat32 to float
  operator float() const { return to_float(value); }

  // Get raw bits representation of tfloat32
  storage_t raw() const { return value; }

  // Logical operators (!,||,&&) are covered if we can cast to bool
  explicit operator bool() { return to_float(value) != 0.0f; }

  // Unary minus operator overloading
  friend tfloat32 operator-(tfloat32 &lhs) { return tfloat32(-to_float(lhs)); }

// Increment and decrement operators overloading
#define OP(op)                                                                 \
  friend tfloat32 &operator op(tfloat32 &lhs) {                                \
    float f = to_float(lhs.value);                                             \
    lhs.value = from_float(op f);                                              \
    return lhs;                                                                \
  }                                                                            \
  friend tfloat32 operator op(tfloat32 &lhs, int) {                            \
    tfloat32 old = lhs;                                                        \
    operator op(lhs);                                                          \
    return old;                                                                \
  }
  OP(++)
  OP(--)
#undef OP

  // Assignment operators overloading
#define OP(op)                                                                 \
  friend tfloat32 &operator op(tfloat32 &lhs, const tfloat32 &rhs) {           \
    float f = static_cast<float>(lhs);                                         \
    f op static_cast<float>(rhs);                                              \
    return lhs = f;                                                            \
  }                                                                            \
  template <typename T>                                                        \
  friend tfloat32 &operator op(tfloat32 &lhs, const T &rhs) {                  \
    float f = static_cast<float>(lhs);                                         \
    f op static_cast<float>(rhs);                                              \
    return lhs = static_cast<float>(f);                                        \
  }                                                                            \
  template <typename T> friend T &operator op(T &lhs, const tfloat32 &rhs) {   \
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
  friend type operator op(const tfloat32 &lhs, const tfloat32 &rhs) {          \
    return type{static_cast<float>(lhs) op static_cast<float>(rhs)};           \
  }                                                                            \
  template <typename T>                                                        \
  friend type operator op(const tfloat32 &lhs, const T &rhs) {                 \
    return type{static_cast<float>(lhs) op static_cast<float>(rhs)};           \
  }                                                                            \
  template <typename T>                                                        \
  friend type operator op(const T &lhs, const tfloat32 &rhs) {                 \
    return type{static_cast<float>(lhs) op static_cast<float>(rhs)};           \
  }
  OP(tfloat32, +)
  OP(tfloat32, -)
  OP(tfloat32, *)
  OP(tfloat32, /)
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
} // namespace oneapi
} // namespace ext

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
