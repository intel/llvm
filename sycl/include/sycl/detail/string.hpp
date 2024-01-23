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
// the user program.
// One most important issue is different ABIs for std::string before
// C++11-ABI and after C++11-ABI.
// To address this issue, we define sycl::detail::string class.
// There are two occasions that std::string crosses the ABI boundaries.
// Once from the user program to the libsycl side, where we need to
// pass the existing std::string to libsycl. This one uses the member 'str'.
// Another occasion is returning std::string from libsycl to the user program.
// In this case, sycl::detail::string provides a placeholder pointer, ret_str.
// libsycl will allocated memory and assign the pointer to it.
// These two boundary crossing cases can happen in one place.
// For example, an API can pass a std::string as a parameter and return
// a std::string. That's why we need two separate members, 'str' and 'ret_str'.
class string {
  const char *str; // used to send user's owning std::string to libsycl
  char *ret_str =
      nullptr; // set from libsycl to return a std::string to a user program

public:
  string() : str(nullptr), ret_str(nullptr) {}
  string(const char *ptr) : str(ptr) {}
  string(std::string &strn) : str(strn.c_str()) {}
  string(const std::string &strn) : str(strn.c_str()) {}

  bool operator==(const char *st) { return strcmp(str, st) == 0; }

  // Once libsycl passes ret_str, we need to reconstruct std::string
  // to return to the user program.
  std::string marshall() { return std::string(ret_str); }

  std::string marshall() const { return std::string(ret_str); }

  // Libsycl calls this method to reconstruct std::string from 'str'
  void unmarshall(std::string &strn) { strcpy(ret_str, strn.c_str()); }

  void allocate(int size) { ret_str = new char[size]; }

  void deallocate() { delete[] ret_str; }

  const char *getPtr() { return str; }

  const char *getPtr() const { return str; }

  char *getRetPtr() { return ret_str; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
