//==----- simd_view_copy_move_assign.cpp  - DPC++ ESIMD on-device test -----==//
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

// This test checks the behavior of simd_view constructors
// and assignment operators.

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

template <class T> bool test(queue q, std::string str, T funcUnderTest) {
  std::cout << "Testing " << str << " ...\n";
  constexpr unsigned VL = 8;

  int A[VL];
  int B[VL];
  // As a result, A should have first 4 values from B, next 4 values from A.
  int gold[VL] = {0, 1, 2, 3, -4, -5, -6, -7};

  for (unsigned i = 0; i < VL; ++i) {
    A[i] = -i;
    B[i] = i;
  }

  try {
    buffer<int, 1> bufA(A, range<1>(VL));
    buffer<int, 1> bufB(B, range<1>(VL));
    range<1> glob_range{1};

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufA.template get_access<access::mode::read_write>(cgh);
      auto PB = bufB.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for(glob_range, [=](id<1> i) SYCL_ESIMD_KERNEL {
        using namespace sycl::ext::intel::experimental::esimd;
        unsigned int offset = i * VL * sizeof(int);
        simd<int, VL> va;
        va.copy_from(PA, offset);
        simd<int, VL> vb;
        vb.copy_from(PB, offset);
        auto va_view = va.select<4, 1>(0);
        auto vb_view = vb.select<4, 1>(0);

        funcUnderTest(va_view, vb_view);

        va.copy_to(PA, offset);
        vb.copy_to(PB, offset);
      });
    });
    q.wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false; // not success
  }

  int err_cnt = 0;
  for (unsigned i = 0; i < VL; ++i) {
    if (A[i] != gold[i]) {
      err_cnt++;
      std::cout << "failed at index " << i << ": " << A[i] << " != " << gold[i]
                << " (gold)\n";
    }
  }

  if (err_cnt > 0) {
    std::cout << "  pass rate: " << ((float)(VL - err_cnt) / (float)VL) * 100.0f
              << "% (" << (VL - err_cnt) << "/" << VL << ")\n";
  }

  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  return err_cnt > 0 ? false : true;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
  // copy constructor creates the same view of the underlying data.
  passed &= test(q, "copy constructor",
                 [](auto &va_view, auto &vb_view) SYCL_ESIMD_FUNCTION {
                   auto va_view_copy(va_view);
                   auto vb_view_copy(vb_view);
                   va_view_copy = vb_view_copy;
                 });
  // move constructor transfers the same view of the underlying data.
  passed &= test(q, "move constructor",
                 [](auto &va_view, auto &vb_view) SYCL_ESIMD_FUNCTION {
                   auto va_view_move(std::move(va_view));
                   auto vb_view_move(std::move(vb_view));
                   va_view_move = vb_view_move;
                 });
  // assignment operator copies the underlying data.
  passed &= test(q, "assignment operator",
                 [](auto &va_view, auto &vb_view)
                     SYCL_ESIMD_FUNCTION { va_view = vb_view; });
  // move assignment operator copies the underlying data.
  passed &= test(q, "move assignment operator",
                 [](auto &va_view, auto &vb_view)
                     SYCL_ESIMD_FUNCTION { va_view = std::move(vb_view); });

  return passed ? 0 : 1;
}
