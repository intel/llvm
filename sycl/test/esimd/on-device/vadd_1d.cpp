//==---------------- vadd_1d.cpp  - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

int main(void) {
  constexpr unsigned Size = 1024 * 128;
  constexpr unsigned VL = 16;

  float *A = new float[Size];
  float *B = new float[Size];
  float *C = new float[Size];

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = 0.0f;
  }

  {
    buffer<float, 1> bufa(A, range<1>(Size));
    buffer<float, 1> bufb(B, range<1>(Size));
    buffer<float, 1> bufc(C, range<1>(Size));

    // We need that many workgroups
    cl::sycl::range<1> GlobalRange{Size / VL};

    // We need that many threads in each group
    cl::sycl::range<1> LocalRange{1};

    queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read>(cgh);
      auto PB = bufb.get_access<access::mode::read>(cgh);
      auto PC = bufc.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class Test>(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::INTEL::gpu;
            unsigned int offset = i * VL * sizeof(float);
            simd<float, VL> va = block_load<float, VL>(PA, offset);
            simd<float, VL> vb = block_load<float, VL>(PB, offset);
            simd<float, VL> vc = va + vb;
            block_store(PC, offset, vc);
          });
    });
    e.wait();
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

  delete[] A;
  delete[] B;
  delete[] C;

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
