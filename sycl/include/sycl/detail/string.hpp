//==----------------- string.hpp - SYCL standard header file ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <string>

#pragma once

namespace sycl {
inline namespace _V1 {
namespace detail {

// This class and detail::string_view class are intended to support
// different ABIs between libsycl and the user program.
// This class is not inteded to replace std::string for general purpose usage.
class string {
  char *str = nullptr;

public:
  string() noexcept = default;
  ~string() { delete[] str; }

  string(std::string_view strn) {
    size_t len = strn.length();
    str = new char[len + 1];
    strn.copy(str, len);
    str[len] = 0;
  }

  friend void swap(string &lhs, string &rhs) noexcept {
    std::swap(lhs.str, rhs.str);
  }

  string(string &&other) noexcept { swap(*this, other); }
  string(const string &other) {
    if (other.str == nullptr)
      return;
    *this = string{other.str};
  }

  string &operator=(string &&other) noexcept {
    swap(*this, other);
    return *this;
  }
  string &operator=(const string &other) {
    *this = string{other};
    return *this;
  }

  string &operator=(const std::string_view strn) {
    *this = string{strn};
    return *this;
  }

  const char *c_str() const noexcept { return str; }

  friend bool operator==(const string &lhs,
                         const std::string_view rhs) noexcept {
    return rhs == lhs.c_str();
  }
  friend bool operator==(const std::string_view lhs,
                         const string &rhs) noexcept {
    return lhs == rhs.c_str();
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
