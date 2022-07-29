//==--------------- replicate_bug.cpp  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// This is a regression test for a bug in simd::replicate_w interface:
// call v.replicate_w<VL, 1>(1) should broadcast second element of v VL times.

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
constexpr int VL = 8;

template <int Width> bool test(queue q, const std::vector<int> &gold) {
  std::cout << "Testing Width=" << Width << " ...\n";
  int *A = malloc_shared<int>(VL * Width, q);
  int *B = malloc_shared<int>(VL * Width, q);

  for (int i = 0; i < VL * Width; ++i) {
    A[i] = i;
    B[i] = -i;
  }

  try {
    range<1> glob_range{1};
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for(glob_range, [=](id<1> i) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::esimd;
        simd<int, VL> va;
        va.copy_from(A);
        simd<int, VL *Width> vb = va.replicate_w<VL, Width>(1);
        vb.copy_to(B);
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    free(B, q);
    return false; // not success
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < VL * Width; ++i) {
    if (B[i] != gold[i]) {
      err_cnt++;
      std::cout << "failed at index " << i << ": " << B[i] << " != " << gold[i]
                << " (gold)\n";
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(VL * Width - err_cnt) / (float)VL * Width) * 100.0f
              << "% (" << (VL * Width - err_cnt) << "/" << VL << ")\n";
  }

  free(A, q);
  free(B, q);

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  passed &= test<1>(q, {1, 1, 1, 1, 1, 1, 1, 1});
  passed &= test<2>(q, {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2});
  return passed ? 0 : 1;
}
