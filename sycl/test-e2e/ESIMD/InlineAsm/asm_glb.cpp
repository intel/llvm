//==---------------- asm_glb.cpp  - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Force -O2 as it currently fails in O0 due to missing VC support
// RUN: %{build} -O2 -o %t.out
// RUN: %{run} %t.out

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

ESIMD_PRIVATE ESIMD_REGISTER(0) simd<float, 16> va;
ESIMD_PRIVATE ESIMD_REGISTER(0) simd<float, 16> vc;

int main(void) {
  constexpr unsigned Size = 1024 * 128;
  constexpr unsigned VL = 16;

  std::vector<float> A(Size);
  std::vector<float> B(Size);
  std::vector<float> C(Size);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = 0.0f;
  }

  buffer<float, 1> bufa(A.data(), A.size());
  buffer<float, 1> bufb(B.data(), B.size());
  buffer<float, 1> bufc(C.data(), C.size());

  try {
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
            va.copy_from(PA, offset);
            simd<float, VL> vb;
            vb.copy_from(PB, offset);
#ifdef __SYCL_DEVICE_ONLY__
            __asm__("add (M1, 16) %0 %1 %2"
                    : "=r"(vc.data_ref())
                    : "r"(va.data()), "r"(vb.data()));
#else
                    vc = va+vb;
#endif
            vc.copy_to(PC, offset);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    return 1;
  }

  sycl::host_accessor A_acc(bufa);
  sycl::host_accessor B_acc(bufb);
  sycl::host_accessor C_acc(bufc);
  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A_acc[i] + B_acc[i] != C_acc[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C_acc[i]
                  << " != " << A_acc[i] << " + " << B_acc[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
