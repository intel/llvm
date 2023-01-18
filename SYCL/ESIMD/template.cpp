//==---------------- template.cpp  - DPC++ ESIMD on-device test ------------==//
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

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr unsigned Size = 1024 * 128;
constexpr unsigned VL = 16;

template <typename T>
sycl::event createKernel(sycl::queue &q, buffer<T, 1> &bufa, buffer<T, 1> &bufb,
                         buffer<T, 1> &bufc) {
  // We need that many workgroups
  range<1> GlobalRange{Size / VL};

  // We need that many threads in each group
  range<1> LocalRange{1};
  return q.submit([&](handler &cgh) {
    auto PA = bufa.template get_access<access::mode::read>(cgh);
    auto PB = bufb.template get_access<access::mode::read>(cgh);
    auto PC = bufc.template get_access<access::mode::write>(cgh);
    cgh.parallel_for<class Test>(GlobalRange * LocalRange,
                                 [=](id<1> i) SYCL_ESIMD_KERNEL {
                                   using namespace sycl::ext::intel::esimd;
                                   unsigned int offset = i * VL * sizeof(T);
                                   simd<T, VL> va;
                                   va.copy_from(PA, offset);
                                   simd<T, VL> vb;
                                   vb.copy_from(PB, offset);
                                   simd<T, VL> vc = va + vb;
                                   vc.copy_to(PC, offset);
                                 });
  });
}

int main(void) {

  float *A = new float[Size];
  float *B = new float[Size];
  float *C = new float[Size];

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = 0.0f;
  }

  try {

    buffer<float, 1> bufa(A, range<1>(Size));
    buffer<float, 1> bufb(B, range<1>(Size));
    buffer<float, 1> bufc(C, range<1>(Size));

    queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto e = createKernel(q, bufa, bufb, bufc);
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    delete[] A;
    delete[] B;
    delete[] C;
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

  delete[] A;
  delete[] B;
  delete[] C;

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
