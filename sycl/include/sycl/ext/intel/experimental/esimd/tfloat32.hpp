//==--------- tfloat32.hpp ------- SYCL tensorfloat32 conversion ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of SIMD tfloat32 type.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/bit_cast.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

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
};

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext

} // namespace _V1
} // namespace sycl
