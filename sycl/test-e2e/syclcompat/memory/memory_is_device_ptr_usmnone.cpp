// ====------ memory_is_device_ptr.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define SYCLCOMPAT_USM_LEVEL_NONE
#include <sycl/detail/core.hpp>
#include <syclcompat/memory.hpp>

int main() {
  float* f = (float*)syclcompat::malloc(sizeof(float));
  bool pass = false;

  if (syclcompat::is_device_ptr(f)) {
    pass = true;
  }

  syclcompat::free(f);

  return (pass ? 0 : 1);
}
