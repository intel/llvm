//==-------------- half_type.hpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/half_type_impl.hpp>

// For std::hash, seems to be the most lightweight header provide it under
// C++17:
#include <optional>

#ifdef __SYCL_DEVICE_ONLY__
#include <iosfwd>
#else
#include <sycl/detail/iostream_proxy.hpp>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail::half_impl {

#ifdef __SYCL_DEVICE_ONLY__
// std::istream/std::ostream aren't usable on device, so don't provide a
// definition to save compile time by using lightweight `<iosfwd>`.
std::ostream &operator<<(std::ostream &O, sycl::half const &rhs);
std::istream &operator>>(std::istream &I, sycl::half &rhs);
#else
inline std::ostream &operator<<(std::ostream &O, sycl::half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

inline std::istream &operator>>(std::istream &I, sycl::half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}
#endif

} // namespace detail::half_impl
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

} // namespace std
