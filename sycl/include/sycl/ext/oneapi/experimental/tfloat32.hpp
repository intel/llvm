//==--------- tfloat32.hpp ------- SYCL tnsorfloat32 conversion
//----------------==//
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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

class tfloat32 {
  using storage_t = float;
  storage_t value;

public:
  tfloat32() = default;
  tfloat32(const bfloat16 &) = default;
  ~tfloat32() = default;

  // Explicit conversion functions
  static storage_t from_float(const float &a) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    int32_t tmp_int = __nvvm_f2tf32_rna(a);
    return __nvvm_bitcast_i2f(tmp_int);
#else
    uint32_t tmp_uint = reinterpret_cast<uint32_t &>(a);
    tmp_uint += 0x1000u;
    tmp_uint &= 0xFFFFE000u;
    float ret = reinterpret_cast<float &>(tmp_uint);
    return ret;
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  }
  static float to_float(const storage_t &a) { return a; }

  static tfloat32 from_bits(const storage_t &a) {
    tfloat32 res;
    res.value = a;
    return res;
  }

  // Implicit conversion from float to tfloat32
  tfloat32(const float &a) { value = from_float(a); }

  tfloat32 &operator=(const float &rhs) {
    value = from_float(rhs);
    return *this;
  }

  // Implicit conversion from bfloat16 to float
  operator float() const { return to_float(value); }
  operator sycl::half() const { return to_float(value); }

  // Get raw bits representation of bfloat16
  storage_t raw() const { return value; }

  // Logical operators (!,||,&&) are covered if we can cast to bool
  explicit operator bool() { return to_float(value) != 0.0f; }

  // Unary minus operator overloading
  friend tfloat32 operator-(bfloat16 &lhs) { return tfloat32(-to_float(lhs)); }

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
    return lhs = f;                                                            \
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
