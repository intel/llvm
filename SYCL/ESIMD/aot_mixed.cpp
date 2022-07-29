//==---------------- aot_mixed.cpp  - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// UNSUPPORTED: esimd_emulator
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device gen9" -o %t.sycl.out -DENABLE_SYCL=0 %s
// RUN: %GPU_RUN_PLACEHOLDER %t.sycl.out
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device gen9" -o %t.out %s
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks the following ESIMD ahead-of-time compilation scenarios:
// 1) When the application contains both SYCL and ESIMD kernel, thus requiring
//    different GPU back-ends (scalar and vector) to kick-in at compile-time.
// 2) When the application contains only ESIMD kernel.

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

#ifndef ENABLE_SIMD
#define ENABLE_SIMD 1
#endif

#ifndef ENABLE_SYCL
#define ENABLE_SYCL 1
#endif

bool verify(float *A, float *B, float *C, size_t Size) {
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
  return err_cnt == 0;
}

constexpr unsigned Size = 1024 * 128;
constexpr unsigned VL = 16;

#if ENABLE_SIMD
bool test_esimd(queue q) {
  std::cout << "Running ESIMD kernel...\n";
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

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read>(cgh);
      auto PB = bufb.get_access<access::mode::read>(cgh);
      auto PC = bufc.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class TestESIMD>(
          Size / VL, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            unsigned int offset = i * VL * sizeof(float);
            simd<float, VL> va;
            va.copy_from(PA, offset);
            simd<float, VL> vb;
            vb.copy_from(PB, offset);
            simd<float, VL> vc = va + vb;
            vc.copy_to(PC, offset);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    delete[] A;
    delete[] B;
    delete[] C;

    return false;
  }
  bool passed = verify(A, B, C, Size);

  delete[] A;
  delete[] B;
  delete[] C;
  return passed;
}
#endif

#if ENABLE_SYCL
bool test_sycl(queue q) {
  std::cout << "Running SYCL kernel...\n";
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

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read>(cgh);
      auto PB = bufb.get_access<access::mode::read>(cgh);
      auto PC = bufc.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class TestSYCL>(Size,
                                       [=](id<1> i) { PC[i] = PA[i] + PB[i]; });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    delete[] A;
    delete[] B;
    delete[] C;

    return false;
  }
  bool passed = verify(A, B, C, Size);

  delete[] A;
  delete[] B;
  delete[] C;
  return passed;
}
#endif

int main(void) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
#if ENABLE_SIMD
  passed &= test_esimd(q);
#endif
#if ENABLE_SYCL
  passed &= test_sycl(q);
#endif

  std::cout << (passed ? "TEST Passed\n" : "TEST FAILED\n");
  return passed ? 0 : 1;
}
