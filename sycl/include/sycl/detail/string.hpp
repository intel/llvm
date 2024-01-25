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

// This class is intended to support different ABIs between libsycl and
// the user program. This class is not inteded to replace std::string
// for general purpose usage.
// One most important issue is different ABIs for std::string before
// C++11-ABI and after C++11-ABI.
// To address this issue, we define sycl::detail::string class.
// Note that there is a separate class detai::string_view for non-owning case.
// There are two occasions that std::string crosses the ABI boundaries.
// Once from the user program to the libsycl side, where we need to
// pass the existing std::string to libsycl. This case should use
// detail::string_view. Another occasion is returning std::string from libsycl
// to the user program. In this case, sycl::detail::string provides a
// placeholder pointer, str. libsycl will allocated memory and assign the
// pointer to it. When this object is returned to sycl.hpp, it will reconstruct
// std::string to return to the user program. These two boundary crossing cases
// can happen in one place. For example, an API can pass a std::string as a
// parameter and return a std::string. That's why we need two separate classes,
// string and string_view.
class string {
  char *str =
      nullptr; // set from libsycl to return a std::string to a user program

public:
  string() = default;

  string(const std::string &strn) {
    allocate(strn.length() + 1);
    strcpy(str, strn.c_str());
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

  ~string() { delete[] str; }

  const char *c_str() { return str; }

  friend bool operator==(string &lhs, const std::string &rhs) {
    // return strcmp(lhs.c_str(), rhs.c_str()) == 0;
    return rhs == lhs.c_str();
  }
  friend bool operator==(const std::string &lhs, string &rhs) {
    // return strcmp(lhs.c_str(), rhs.c_str()) == 0;
    return lhs == rhs.c_str();
  }

private:
  void allocate(int size) {
    // This class object is immutable once its memory is allocated.
    // Reallocation is not allowed.
    assert(str == nullptr &&
           "Error: memory already allocated for this object.");
    str = new char[size];
  } // called by libsycl before returning
};

} // namespace detail
} // namespace _V1
} // namespace sycl
