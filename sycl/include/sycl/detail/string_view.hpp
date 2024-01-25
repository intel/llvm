//==----------------- string_view.hpp - SYCL standard header file
//---------------==//
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

// This class  and detail::string are intended to support different ABIs
// between libsycl and the user program. This class is not inteded to replace
// std::string for general purpose usage.
// One most important issue is different ABIs for std::string before
// C++11-ABI and after C++11-ABI.
// To address this issue, we define sycl::detail::string_view class.
// There are two occasions that std::string crosses the ABI boundaries.
// Once from the user program to the libsycl side, where we need to
// pass the existing std::string to libsycl. This one uses the member 'str'.
// Another occasion is returning std::string from libsycl to the user program.
// In this case, sycl::detail::string provides a placeholder pointer, str.
// libsycl will allocated memory and assign the pointer to it.
// These two boundary crossing cases can happen in one place.
// For example, an API can pass a std::string as a parameter and return
// a std::string. That's why we need two separate classes, string and
// string_view.
class string_view {
  const char *str =
      nullptr; // used to send user's owning std::string to libsycl

public:
  string_view() = default;

  string_view(std::string &strn) : str(strn.c_str()) {}
  string_view(const std::string &strn) : str(strn.c_str()) {}

  string_view(string_view &&strn) : str(strn.str) {}
  string_view(const string_view &strn) : str(strn.str) {}

  string_view &operator=(string_view &&strn) {
    str = strn.str;
    return *this;
  }
  string_view &operator=(const string_view &strn) {
    str = strn.str;
    return *this;
  }

  const char *c_str() { return str; }
  const char *c_str() const { return str; }

  friend bool operator==(string_view &lhs, const std::string &rhs) {
    // return strcmp(lhs.c_str(), rhs.c_str()) == 0;
    return rhs == lhs.c_str();
  }
  friend bool operator==(const std::string &lhs, string_view &rhs) {
    // return strcmp(lhs.c_str(), rhs.c_str()) == 0;
    return lhs == rhs.c_str();
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
