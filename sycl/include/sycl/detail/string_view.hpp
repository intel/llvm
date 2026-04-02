//==-------------- string_view.hpp - SYCL standard header file -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/string.hpp>

#include <string>

namespace sycl {
inline namespace _V1 {
namespace detail {

// This class and detail::string class are intended to support
// different ABIs between libsycl and the user program.
// This class is not intended to replace std::string_view for general purpose
// usage.

class string_view {
  const char *str = nullptr;
  size_t len = 0;

public:
  constexpr string_view() noexcept = default;
  constexpr string_view(const string_view &strn) noexcept = default;
  constexpr string_view(string_view &&strn) noexcept = default;
  constexpr string_view(std::string_view strn) noexcept
      : str(strn.data()), len(strn.size()) {}
  string_view(const sycl::detail::string &strn) noexcept
      : str(strn.c_str()), len(strlen(strn.c_str())) {}

  constexpr string_view &operator=(string_view &&strn) noexcept = default;
  string_view &operator=(const string_view &strn) noexcept = default;

  constexpr string_view &operator=(std::string_view strn) noexcept {
    str = strn.data();
    len = strn.size();
    return *this;
  }

  string_view &operator=(const sycl::detail::string &strn) noexcept {
    str = strn.c_str();
    len = strlen(strn.c_str());
    return *this;
  }

  constexpr const char *data() const noexcept { return str ? str : ""; }

  constexpr operator std::string_view() const noexcept {
    if (str == nullptr)
      return std::string_view{};
    return std::string_view(str, len);
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
