//==---------------- vadd_usm.cpp  - DPC++ ESIMD on-device test ------------==//
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

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace cl::sycl;

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned VL = 32;
  constexpr unsigned GroupSize = 8;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  float *A = malloc_shared<float>(Size, q);
  float *B = malloc_shared<float>(Size, q);
  float *C = malloc_shared<float>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  // We need that many workitems. Each processes VL elements of data.
  range<1> GlobalRange{Size / VL};
  // Number of workitems in each workgroup.
  range<1> LocalRange{GroupSize};

  nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
          Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::experimental::esimd;

            int i = ndi.get_global_id(0);
            simd<float, VL> va;
            va.copy_from(A + i * VL);
            simd<float, VL> vb;
            vb.copy_from(B + i * VL);
            simd<float, VL> vc = va + vb;
            vc.copy_to(C + i * VL);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(A, q);
    free(B, q);
    free(C, q);
    return 1;
  }

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " + " << B[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  free(A, q);
  free(B, q);
  free(C, q);
  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
