//==--- accessor_stateless_ctor_64.cpp  - DPC++ ESIMD on-device test---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

int main(void) {
  constexpr unsigned VL = 16;
  constexpr uint64_t Gig = 1024 * 1024 * 1024;

  constexpr uint64_t Size = Gig / 2 + 16;

  uint64_t *A = new uint64_t[Size];

  for (uint64_t i = 0; i < Size; ++i)
    A[i] = i;
  buffer<uint64_t, 1> bufa(A, range<1>(Size));

  try {
    queue q;

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    q.submit([&](handler &cgh) {
       auto PA = bufa.get_access<access::mode::read_write>(cgh);
       cgh.single_task<class Test>([=]() SYCL_ESIMD_KERNEL {
         uint64_t offset = (Size - VL) * sizeof(uint64_t);
         simd<uint64_t, VL> va(PA, 0);
         simd<uint64_t, VL> vb(PA, offset);
         va *= 2;
         vb *= 5;
         va.copy_to(PA, 0);
         vb.copy_to(PA, offset);
       });
     }).wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    delete[] A;
    return 1;
  }

  bool failed = false;
  host_accessor A_acc(bufa);
  for (uint64_t I = 0; I < VL; I++) {
    uint64_t Expected = I * 2;
    if (A_acc[I] != Expected) {
      std::cout << "FAILED: " << A_acc[I] << " != " << Expected << std::endl;
      failed = true;
    }
  }
  for (uint64_t I = Size - VL; I < Size; I++) {
    uint64_t Expected = I * 5;
    if (A_acc[I] != Expected) {
      std::cout << "FAILED: " << A_acc[I] << " != " << Expected << std::endl;
      failed = true;
    }
  }

  if (!failed)
    std::cout << "PASSED" << std::endl;

  delete[] A;

  return failed;
}
