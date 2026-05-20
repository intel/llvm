//==-------------- fp4.hpp - DPC++ Explicit SIMD API ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implementation of SIMD fp4 storage types. These types represent 4 bit
// floating point types for storage purposes and simplification of dpas
// interface. No arithmetic operations with FP4 are supported. The types hold a
// pair of fp4 numbers as a uint8_t. No extraction or conversion operations are
// supported.
//===----------------------------------------------------------------------===//

#pragma once
#include <cstdint> //for uint8_t

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

class fp4_S1E2M1 {
  using storage_t = uint8_t;
  storage_t value;

public:
  fp4_S1E2M1() = default;
  fp4_S1E2M1(const fp4_S1E2M1 &) = default;
  ~fp4_S1E2M1() = default;

  fp4_S1E2M1(storage_t val) : value(val) {}

  fp4_S1E2M1 &operator=(const storage_t &rhs) {
    value = rhs;
    return *this;
  }

  // Get raw bits representation of fp4_S1E2M1
  storage_t raw() const { return value; }
  bool operator==(const fp4_S1E2M1 &rhs) { return value == rhs.raw(); }
  bool operator!=(const fp4_S1E2M1 &rhs) { return value != rhs.raw(); }
  operator uint8_t() const { return value; }
};

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext

} // namespace _V1
} // namespace sycl
