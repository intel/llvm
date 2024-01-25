//==-------------- string_view.hpp - SYCL standard header file -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
inline namespace _V1 {
namespace detail {

// This class and detail::string class are intended to support
// different ABIs between libsycl and the user program.
// This class is not inteded to replace std::string_view for general purpose
// usage.
class string_view {
  const char *str =
      nullptr; // used to send user's owning std::string to libsycl

public:
  string_view() = default;
  string_view(const string_view &strn) = default;
  string_view(const std::string_view &strn) : str(strn.data()) {}
  string_view(string_view &&strn) : str(strn.data()) {}

  string_view &operator=(string_view &&strn) {
    str = strn.str;
    return *this;
  }
  string_view &operator=(const string_view &strn) {
    str = strn.str;
    return *this;
  }

  string_view &operator=(string &&strn) {
    str = strn.c_str();
    return *this;
  }
  string_view &operator=(const string &strn) {
    str = strn.c_str();
    return *this;
  }

  string_view &operator=(const std::string_view &strn) {
    str = strn.data();
    return *this;
  }

  const char *data() { return str; }
  const char *data() const { return str; }

  friend bool operator==(const string_view &lhs, const std::string_view &rhs) {
    return rhs == lhs.data();
  }
  friend bool operator==(const std::string_view &lhs, const string_view &rhs) {
    return lhs == rhs.data();
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
