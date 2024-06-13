//==-acc_gather_scatter_rgba_stateless_64.cpp - DPC++ ESIMD on-device test-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------==//
// REQUIRES: gpu-intel-pvc
// RUN: %{build} -fsycl-esimd-force-stateless-mem -o %t.out
// RUN: %{run} %t.out
//
// The test checks functionality of the gather_rgba/scatter_rgba accessor-based
// ESIMD intrinsics when stateless memory accesses are enforced, i.e. accessor
// based accesses are automatically converted to stateless accesses with 64-bit
// offsets.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>

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
  queue q;

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  try {
    q.submit([&](handler &cgh) {
       auto PA = bufa.get_access<access::mode::read_write>(cgh);
       cgh.single_task<class Test>([=]() SYCL_ESIMD_KERNEL {
         uint64_t offsetStart = (Size - VL) * sizeof(uint64_t);
         simd<uint64_t, VL> offset(offsetStart, sizeof(uint64_t));
         simd<uint64_t, VL> beginning(0, sizeof(uint64_t));
         simd<uint32_t, VL> va =
             gather_rgba<rgba_channel_mask::R, decltype(PA), VL, uint32_t>(
                 PA, beginning);
         simd<uint32_t, VL> vb =
             gather_rgba<rgba_channel_mask::R, decltype(PA), VL, uint32_t>(
                 PA, offset);
         va *= 2;
         vb *= 5;
         scatter_rgba<rgba_channel_mask::R, decltype(PA), VL, uint32_t>(
             PA, beginning, va);
         scatter_rgba<rgba_channel_mask::R, decltype(PA), VL, uint32_t>(
             PA, offset, vb);
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
    uint64_t Expected = static_cast<uint32_t>(I) * 2;
    if (A_acc[I] != Expected) {
      std::cout << "FAILED: " << A_acc[I] << " != " << Expected << std::endl;
      failed = true;
    }
  }
  for (uint64_t I = Size - VL; I < Size; I++) {
    uint64_t Expected = static_cast<uint32_t>(I) * 5;
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
