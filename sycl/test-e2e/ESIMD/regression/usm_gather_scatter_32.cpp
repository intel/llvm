//==---------- usm_gather_scatter_32.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -I%S/.. -o %t.out
// RUN: %{run} %t.out

// Regression test for checking USM-based gather/scatter with 32 elements.

#include "../esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

using DataT = std::uint8_t;

int main(void) {
  constexpr unsigned Size = 1024;
  constexpr unsigned VL = 32;

  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

  DataT *A = malloc_shared<DataT>(Size, q);
  DataT *C = malloc_shared<DataT>(Size, q);

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = i;
  }

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>({Size / VL},
                                 [=](item<1> ndi) SYCL_ESIMD_KERNEL {
                                   int i = ndi.get_id(0);
                                   simd<DataT, VL> va;
                                   va.copy_from(A + i * VL);

                                   simd<uint32_t, VL> offsets(0, 1);
                                   scatter<DataT, VL>(C + i * VL, offsets, va);
                                 });
  });
  e.wait();

  std::cout << "Elements should be equal to indices after scatter operation\n";
  std::cout << "C[16] C[17] C[18] C[19]:\n";
  std::cout << +C[16] << " " << +C[17] << " " << +C[18] << " " << +C[19]
            << "\n";
  int err_cnt = 0;

  for (int i = 0; i < Size; i++) {
    if ((C[i] != (DataT)i) && (++err_cnt < VL)) {
      std::cerr << "### ERROR at " << i << ": " << C[i] << " != " << (int)i
                << "(gold)\n";
    }
  }
  sycl::free(A, q);
  sycl::free(C, q);

  if (err_cnt == 0) {
    std::cout << "OK\n";
    return 0;
  } else {
    std::cout << "FAIL\n";
    return 1;
  }
}
