//=---lsc_gather_scatter_stateless_64.cpp - DPC++ ESIMD on-device test---=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=----------------------------------------------------------------------=//
// REQUIRES: arch-intel_gpu_pvc
// RUN: %{build} -o %t.out -fsycl-esimd-force-stateless-mem
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

int main() {
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
         simd<uint64_t, VL> va = lsc_gather<uint64_t>(PA, beginning);
         simd_mask<VL> pred = 1;
         simd<uint64_t, VL> old_values = 0;
         lsc_prefetch<uint64_t, 1, lsc_data_size::default_size,
                      cache_hint::cached, cache_hint::cached>(PA, offset);
         simd<uint64_t, VL> vb =
             lsc_gather<uint64_t>(PA, offset, pred, old_values);
         simd<uint64_t, VL> vc = lsc_gather<uint64_t>(PA, offset);
         va *= 5;
         vb += vc;
         lsc_scatter<uint64_t>(PA, beginning, va);
         lsc_scatter<uint64_t>(PA, offset, vb);
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
    uint64_t Expected = static_cast<uint32_t>(I) * 5;
    if (A_acc[I] != Expected) {
      std::cout << "FAILED: " << A_acc[I] << " != " << Expected << std::endl;
      failed = true;
    }
  }
  for (uint64_t I = Size - VL; I < Size; I++) {
    uint64_t Expected = static_cast<uint32_t>(I) * 2;
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
