//==----------------- string.hpp - SYCL standard header file ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>

#pragma once

namespace sycl {
inline namespace _V1 {

class string {
  const char *str; // used to send existing std::string to libsycl
  char *ret_str;   // set from libsycl

public:
  string() : str(nullptr), ret_str(nullptr) {}
  string(const char *ptr) : str(ptr) {}
  string(std::string strn) : str(strn.c_str()) {}

  bool operator==(const char *st) { return strcmp(str, st) == 0; }

  std::string marshall() { return std::string(ret_str); }

  std::string marshall() const { return std::string(ret_str); }

  void unmarshall(std::string &strn) { strcpy(ret_str, strn.c_str()); }

  void allocate(int size) { ret_str = new char[size]; }

  void deallocate() { delete[] ret_str; }

  const char *getPtr() { return str; }

  char *getRetPtr() { return ret_str; }
};

} // namespace _V1
} // namespace sycl