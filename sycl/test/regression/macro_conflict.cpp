// RUN: %clangxx -fsycl %s -o %t.out                                                                     
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// expected-no-diagnostics
//
//==-------------- macro_conflict.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks if the user-defined macros SUCCESS, FAIL, BLOCKED are
// defined in global namespace by sycl.hpp
//===----------------------------------------------------------------------===//

#define SUCCESS 0
#define FAIL 1
#define BLOCKED 2

#include <CL/sycl.hpp>

int main() {
  printf("hello world!\n");
  return 0;
}
