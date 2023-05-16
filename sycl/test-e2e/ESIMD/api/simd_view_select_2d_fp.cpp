//==------- simd_view_select_2d_fp.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// Smoke test for 2D region select API which can be used to represent 2D tiles.
// Tests FP types.

#include "simd_view_select_2d.hpp"

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  bool passed = true;
  if (dev.has(sycl::aspect::fp16))
    passed &= test<half>(q);
  passed &= test<float>(q);
  if (dev.has(sycl::aspect::fp64))
    passed &= test<double>(q);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
