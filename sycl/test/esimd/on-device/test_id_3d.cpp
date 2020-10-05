//==---------------- test_id_3d.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

// This test checks that multi-dimensional sycl::item can be used in ESIMD
// kernels.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

int main(void) {
  constexpr unsigned LocalSizeX = 32;
  constexpr unsigned VL = 8;
  sycl::range<3> GroupRange{2, 3, 4};
  sycl::range<3> LocalRange{2, 3, LocalSizeX / VL};
  sycl::range<3> GlobalRange = GroupRange * LocalRange;
  sycl::range<3> ScalarGlobalRange = GlobalRange * sycl::range<3>{1, 1, VL};
  size_t Y = GlobalRange[1];
  size_t X = GlobalRange[2];

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();
  int *C = static_cast<int *>(
      malloc_shared(ScalarGlobalRange.size() * sizeof(int), dev, ctxt));

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(
        GlobalRange, [=](item<3> it) SYCL_ESIMD_KERNEL {
          using namespace sycl::INTEL::gpu;
          auto id = it.get_id();
          // calculate linear ID:
          size_t lin_id = id[0] * Y * X + id[1] * X + id[2];
          simd<int, VL> inc(0, 1);
          int off = (int)(lin_id * VL);
          simd<int, VL> val = inc + off;
          block_store<int, VL>(C + off, val);
        });
  });
  e.wait();
  int err_cnt = 0;

  for (size_t i = 0; i < ScalarGlobalRange.size(); ++i) {
    if (C[i] != (int)i) {
      if (++err_cnt < 10) {
        std::cerr << "*** ERROR at " << i << ": " << C[i] << "!=" << i
                  << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "FAILED. " << err_cnt << "/" << ScalarGlobalRange.size()
              << " errors\n";
    return 1;
  }
  std::cout << "passed\n";
  return 0;
}
