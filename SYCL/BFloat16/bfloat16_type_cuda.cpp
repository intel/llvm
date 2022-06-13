// REQUIRES: gpu, cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 %s -o %t.out
// RUN: %t.out

//==--------- bfloat16_type_cuda.cpp - SYCL bfloat16 type test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bfloat16_type.hpp"

int main() {
  bool has_bfloat16_aspect = false;
  for (const auto &plt : sycl::platform::get_platforms()) {
    if (plt.has(aspect::ext_oneapi_bfloat16))
      has_bfloat16_aspect = true;
  }

  if (has_bfloat16_aspect)
    return run_tests();
}
