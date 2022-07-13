//==---------------- accessor.cpp  - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -D_CRT_SECURE_NO_WARNINGS=1 %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks that accessor-based memory accesses work correctly in ESIMD.

#include "esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

constexpr unsigned int VL = 1024 * 128;

using Ty = float;

int main() {
  Ty *data0 = new float[VL];
  Ty *data1 = new float[VL];
  constexpr Ty VAL = 5;

  for (int i = 0; i < VL; i++) {
    data0[i] = i;
  }

  try {
    queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

    buffer<Ty, 1> buf0(data0, range<1>(VL));
    buffer<Ty, 1> buf1(data1, range<1>(VL));

    q.submit([&](handler &cgh) {
      std::cout << "Running on "
                << q.get_device().get_info<cl::sycl::info::device::name>()
                << "\n";

      auto acc0 = buf0.get_access<access::mode::read_write>(cgh);
      auto acc1 = buf1.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class Test>(range<1>(1),
                                   [=](sycl::id<1> i) SYCL_ESIMD_KERNEL {
                                     using namespace sycl::ext::intel::esimd;
                                     unsigned int offset = 0;
                                     for (int k = 0; k < VL / 16; k++) {
                                       simd<Ty, 16> var;
                                       var.copy_from(acc0, offset);
                                       var += VAL;
                                       var.copy_to(acc0, offset);
                                       var += 1;
                                       var.copy_to(acc1, offset);
                                       offset += 64;
                                     }
                                   });
    });

    q.wait();

  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  int err_cnt = 0;

  for (int i = 0; i < VL; i++) {
    Ty gold0 = i + VAL;
    Ty gold1 = gold0 + 1;
    Ty val0 = data0[i];
    Ty val1 = data1[i];

    if (val0 != gold0) {
      if (++err_cnt < 10)
        std::cerr << "*** ERROR at data0[" << i << "]: " << val0
                  << " != " << gold0 << "(gold)\n";
    }
    if (val1 != gold1) {
      if (++err_cnt < 10)
        std::cerr << "*** ERROR at data1[" << i << "]: " << val1
                  << " != " << gold1 << "(gold)\n";
    }
  }

  delete[] data0;
  delete[] data1;

  if (err_cnt == 0) {
    std::cout << "Passed\n";
    return 0;
  } else {
    std::cout << "Failed: " << err_cnt << " of " << VL << " errors\n";
    return 1;
  }
}
