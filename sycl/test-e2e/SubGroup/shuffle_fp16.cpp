// REQUIRES: aspect-fp16
// REQUIRES: gpu

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip
// Even though `gfx908` and `gfx906` support halfs, libspirv is currently
// built with `tahiti` as the target CPU, which means that clang rejects
// AMD built-ins using halfs, for that reason half support has to stay
// disabled.

//==------- shuffle_fp16.cpp - SYCL sub_group shuffle test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shuffle.hpp"
#include <iostream>

int main() {
  queue Queue;
  check<half>(Queue);
  check<half, 2>(Queue);
  check<half, 4>(Queue);
  check<half, 8>(Queue);
  check<half, 16>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
