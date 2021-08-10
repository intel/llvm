//==----- simd_subscript_operator.cpp  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || rocm
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks that it's possible to write through the simd subscript
// operator. E.g.:
//   simd<int, 4> v = 1;
//   v[1] = 0; // v[1] returns writable simd_view

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  constexpr unsigned VL = 16;

  int A[VL];
  int B[VL];
  int gold[VL];

  for (unsigned i = 0; i < VL; ++i) {
    A[i] = -i;
    B[i] = i;
    gold[i] = B[i];
  }

  // some random indices to overwrite elements in B with elements from A.
  std::array<int, 5> indicesToCopy = {2, 5, 9, 10, 13};

  try {
    buffer<int, 1> bufA(A, range<1>(VL));
    buffer<int, 1> bufB(B, range<1>(VL));
    range<1> glob_range{1};

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufA.get_access<access::mode::read>(cgh);
      auto PB = bufB.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class Test>(glob_range, [=](id<1> i) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::experimental::esimd;
        unsigned int offset = i * VL * sizeof(int);
        simd<int, VL> va;
        va.copy_from(PA, offset);
        simd<int, VL> vb;
        vb.copy_from(PB, offset);
        for (auto idx : indicesToCopy)
          vb[idx] = va[idx];
        vb.copy_to(PB, offset);
      });
    });
    q.wait_and_throw();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.get_cl_code();
  }

  int err_cnt = 0;

  for (auto i : indicesToCopy)
    gold[i] = A[i];

  for (unsigned i = 0; i < VL; ++i) {
    int val = B[i];

    if (val != gold[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ": " << val << " != " << gold[i]
                  << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: " << ((float)(VL - err_cnt) / (float)VL) * 100.0f
              << "% (" << (VL - err_cnt) << "/" << VL << ")\n";
  }

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");

  return err_cnt > 0 ? 1 : 0;
}
