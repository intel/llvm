//==---------------- test_id_3d.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks that multi-dimensional sycl::item can be used in ESIMD
// kernels.

#include "esimd_test_utils.hpp"

using namespace sycl;

int main(void) {
  constexpr unsigned LocalSizeX = 32;
  constexpr unsigned VL = 8;
  sycl::range<3> GroupRange{2, 3, 4};
  sycl::range<3> LocalRange{2, 3, LocalSizeX / VL};
  sycl::range<3> GlobalRange = GroupRange * LocalRange;
  sycl::range<3> ScalarGlobalRange = GlobalRange * sycl::range<3>{1, 1, VL};
  size_t Y = GlobalRange[1];
  size_t X = GlobalRange[2];

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  int *C = malloc_shared<int>(ScalarGlobalRange.size(), q);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
          GlobalRange, [=](item<3> it) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            auto id = it.get_id();
            // calculate linear ID:
            size_t lin_id = id[0] * Y * X + id[1] * X + id[2];
            simd<int, VL> inc(0, 1);
            int off = (int)(lin_id * VL);
            simd<int, VL> val = inc + off;
            val.copy_to(C + off);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(C, q);
    return 1;
  }

  int err_cnt = 0;

  for (size_t i = 0; i < ScalarGlobalRange.size(); ++i) {
    if (C[i] != (int)i) {
      if (++err_cnt < 10) {
        std::cerr << "*** ERROR at " << i << ": " << C[i] << "!=" << i
                  << " (gold)\n";
      }
    }
  }
  free(C, q);
  if (err_cnt > 0) {
    std::cout << "FAILED. " << err_cnt << "/" << ScalarGlobalRange.size()
              << " errors\n";
    return 1;
  }
  std::cout << "passed\n";
  return 0;
}
