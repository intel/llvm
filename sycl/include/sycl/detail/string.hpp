//==----------------- string.hpp - SYCL standard header file ---------------==//
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

// This class and detail::string_view class are intended to support
// different ABIs between libsycl and the user program.
// This class is not inteded to replace std::string for general purpose usage.
class string {
  char *str =
      nullptr; // set from libsycl to return a std::string to a user program

public:
  string() = default;

  string(const std::string_view &strn) {
    allocate(strn.length() + 1);
    strcpy(str, strn.data());
  }

  string(string &&strn) {
    str = strn.str;
    strn.str = nullptr;
  }
  string(const string &strn) {
    allocate(strlen(strn.str) + 1);
    strcpy(str, strn.str);
  }

  string &operator=(string &&strn) {
    delete[] str;
    str = strn.str;
    strn.str = nullptr;
    return *this;
  }
  string &operator=(const string &strn) {
    delete[] str;
    str = nullptr;
    allocate(strlen(strn.str) + 1);
    strcpy(str, strn.str);
    return *this;
  }

  string &operator=(const std::string_view &strn) {
    allocate(strn.length() + 1);
    strcpy(str, strn.data());
    return *this;
  }

  ~string() { delete[] str; }

  const char *c_str() { return str; }
  const char *c_str() const { return str; }

  friend bool operator==(const string &lhs, const std::string_view &rhs) {
    return rhs == lhs.c_str();
  }
  friend bool operator==(const std::string_view &lhs, const string &rhs) {
    return lhs == rhs.c_str();
  }

private:
  void allocate(int size) {
    delete[] str;
    str = new char[size];
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
