//==- operator_assignment_glb_mask.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include "common.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

constexpr unsigned VL = 16;

ESIMD_PRIVATE ESIMD_REGISTER(0) simd_mask<VL> va;
ESIMD_PRIVATE ESIMD_REGISTER(0) simd_mask<VL> vb;

int main(void) {
  constexpr unsigned Size = 1024;

  std::vector<unsigned short> A(Size);
  std::vector<unsigned short> B(Size);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i % std::numeric_limits<unsigned short>::max();
    B[i] = 0;
  }

  buffer<unsigned short, 1> bufa(A.data(), A.size());
  buffer<unsigned short, 1> bufb(B.data(), B.size());

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
      auto PB = bufb.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class Test>(
          GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;
            unsigned int offset = i * VL * sizeof(unsigned short);
            va.copy_from(PA, offset);
            vb = va;
            vb.copy_to(PB, offset);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';

    return 1;
  }

  sycl::host_accessor A_acc(bufa);
  sycl::host_accessor B_acc(bufb);
  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A_acc[i] != B_acc[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << B_acc[i]
                  << " != " << A_acc[i] << "\n";
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
