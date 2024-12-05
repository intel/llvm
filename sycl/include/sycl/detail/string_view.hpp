//==-------------- string_view.hpp - SYCL standard header file -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#pragma once

namespace sycl {
inline namespace _V1 {
namespace detail {

// This class and detail::string class are intended to support
// different ABIs between libsycl and the user program.
// This class is not inteded to replace std::string_view for general purpose
// usage.
class string_view {
  const char *str = nullptr;

public:
  string_view() noexcept = default;
  string_view(const string_view &strn) noexcept = default;
  string_view(string_view &&strn) noexcept = default;
  string_view(std::string_view strn) noexcept : str(strn.data()) {}

  string_view &operator=(string_view &&strn) noexcept = default;
  string_view &operator=(const string_view &strn) noexcept = default;

  string_view &operator=(std::string_view strn) noexcept {
    str = strn.data();
    return *this;
  }

  const char *data() const noexcept { return str; }

  friend bool operator==(const string_view &lhs,
                         std::string_view rhs) noexcept {
    return rhs == lhs.data();
  }
  friend bool operator==(std::string_view lhs,
                         const string_view &rhs) noexcept {
    return lhs == rhs.data();
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
