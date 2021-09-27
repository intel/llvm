//==-------------- half_type.cpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/half_type.hpp>
// This is included to enable __builtin_expect()
#include <detail/platform_util.hpp>

#include <cstring>
#include <iostream>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

namespace host_half_impl {

half::half(const float &RHS) : Buf(float2Half(RHS)) {}

half &half::operator+=(const half &RHS) {
  *this = operator float() + static_cast<float>(RHS);
  return *this;
}

half &half::operator-=(const half &RHS) {
  *this = operator float() - static_cast<float>(RHS);
  return *this;
}

half &half::operator*=(const half &RHS) {
  *this = operator float() * static_cast<float>(RHS);
  return *this;
}

half &half::operator/=(const half &RHS) {
  *this = operator float() / static_cast<float>(RHS);
  return *this;
}

half::operator float() const { return half2Float(Buf); }

// Operator +, -, *, /
half operator+(half LHS, const half &RHS) {
  LHS += RHS;
  return LHS;
}

half operator-(half LHS, const half &RHS) {
  LHS -= RHS;
  return LHS;
}

half operator*(half LHS, const half &RHS) {
  LHS *= RHS;
  return LHS;
}

half operator/(half LHS, const half &RHS) {
  LHS /= RHS;
  return LHS;
}

// Operator <, >, <=, >=
bool operator<(const half &LHS, const half &RHS) {
  return static_cast<float>(LHS) < static_cast<float>(RHS);
}

bool operator>(const half &LHS, const half &RHS) { return RHS < LHS; }

bool operator<=(const half &LHS, const half &RHS) { return !(LHS > RHS); }

bool operator>=(const half &LHS, const half &RHS) { return !(LHS < RHS); }

// Operator ==, !=
bool operator==(const half &LHS, const half &RHS) {
  return static_cast<float>(LHS) == static_cast<float>(RHS);
}

bool operator!=(const half &LHS, const half &RHS) { return !(LHS == RHS); }
} // namespace host_half_impl

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

__SYCL_EXPORT std::ostream &operator<<(std::ostream &O,
                                       cl::sycl::half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

__SYCL_EXPORT std::istream &operator>>(std::istream &I, cl::sycl::half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}
