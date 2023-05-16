//==---------------- asm_simd_view.cpp  - DPC++ ESIMD on-device test
//-------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

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

  try {
    buffer<float, 1> bufa(A, range<1>(Size));
    buffer<float, 1> bufb(B, range<1>(Size));
    buffer<float, 1> bufc(C, range<1>(Size));

    // We need that many workgroups
    range<1> GlobalRange{Size / VL};

    // We need that many threads in each group
    range<1> LocalRange{1};

    queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    auto e = q.submit([&](handler &cgh) {
      auto PA = bufa.get_access<access::mode::read>(cgh);
      auto PB = bufb.get_access<access::mode::read>(cgh);
      auto PC = bufc.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class Test>(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            unsigned int offset = i * VL * sizeof(float);
            simd<float, VL> va;
            va.copy_from(PA, offset);
            simd<float, VL> vb;
            vb.copy_from(PB, offset);
#ifdef __SYCL_DEVICE_ONLY__
            auto va_half1 = va.select<VL / 2, 1>();
            auto va_half2 = va.select<VL / 2, 1>(VL / 2);
            auto vb_half1 = vb.select<VL / 2, 1>();
            auto vb_half2 = vb.select<VL / 2, 1>(VL / 2);
            simd<float, VL / 2> out1;
            simd<float, VL / 2> out2;
            // simd_view is not supported in l-value context in inline asm, so
            // use simd to store the result
            __asm__("add (M1, 8) %0 %1 %2"
                    : "=r"(out1.data_ref())
                    : "r"(va_half1.data()), "r"(vb_half1.data()));
            __asm__("add (M1, 8) %0 %1 %2"
                    : "=r"(out2.data_ref())
                    : "r"(va_half2.data()), "r"(vb_half2.data()));
            out1.copy_to(PC, offset);
            out2.copy_to(PC, offset + ((VL / 2) * sizeof(float)));
#else
             simd<float, VL> vc;
             vc = va+vb;
             vc.copy_to(PC, offset);
#endif
          });
    });
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
