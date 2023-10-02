//==---------------- half-type.cpp - SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/half_type.hpp>

#include <iostream>
#include <sycl/detail/export.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
namespace half_impl {

std::ostream &operator<<(std::ostream &O, half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

std::istream &operator>>(std::istream &I, half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

} // namespace half_impl
} // namespace detail
} // namespace _V1
} // namespace sycl