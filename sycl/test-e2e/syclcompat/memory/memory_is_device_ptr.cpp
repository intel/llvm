// ====------ memory_is_device_ptr.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

int main() {
  float* f = (float*)dpct::dpct_malloc(sizeof(float));
  bool pass = false;

  if (dpct::is_device_ptr(f)) {
    pass = true;
  }

  dpct::dpct_free(f);

  return (pass ? 0 : 1);
}