//==-------------- half_type.hpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>
#include <sycl/detail/half_type_impl.hpp>

// For std::hash, seems to be the most lightweight header provide it under
// C++17:
#include <optional>

#include <iosfwd> // for ostream, istream

namespace sycl {
inline namespace _V1 {
namespace detail::half_impl {

// Stream operators are defined out-of-line in the SYCL runtime. The header
// only carries forward declarations so that <ostream>/<istream> are not
// pulled into device compilation. Consumers that print sycl::half values
// must include <iostream> (or <ostream>/<istream>) themselves.
__SYCL_DEPRECATED(
    "Stream operators for half are deprecated and will be removed in a "
    "future release. Please use explicit conversion to "
    "float for streaming.")
__SYCL_EXPORT std::ostream &operator<<(std::ostream &O, sycl::half const &rhs);
__SYCL_DEPRECATED(
    "Stream operators for half are deprecated and will be removed in a "
    "future release. Please use explicit conversion to "
    "float for streaming.")
__SYCL_EXPORT std::istream &operator>>(std::istream &I, sycl::half &rhs);

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
