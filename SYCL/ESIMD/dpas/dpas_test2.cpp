//==---------------- dpas_test2.cpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || esimd_emulator
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -DESIMD_XE_HPC %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

int main(void) {
  constexpr unsigned Size = 128;
  constexpr unsigned VL = 16;
  constexpr unsigned GroupSize = 1;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int *C = malloc_shared<int>(Size, q);
  memset(C, 0, Size * sizeof(int));

  // We need that many task groups
  range<1> GroupRange{GroupSize};

  // We need that many tasks in each group
  range<1> TaskRange{GroupSize};
  nd_range<1> Range(GroupRange, TaskRange);

  q.submit([&](handler &cgh) {
     cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
       using namespace sycl::ext::intel::esimd;
       using namespace sycl::ext::intel::experimental::esimd;

       simd<char, Size * 2> va(0);
       auto ma = va.bit_cast_view<char, 8, 32>();
       ma.select<2, 1, 8, 4>(0, 0) = 4;

       simd<char, Size> vb(0);
       auto mb = vb.bit_cast_view<char, 8, 16>();
       mb.select<8, 1, 1, 1>(0, 0) = 4;

       simd<int, Size> vc(0);
       vc =
           dpas<argument_type::S2, argument_type::S2, 8, 8, int, int, int, Size,
                64, 32>(vc, ma.bit_cast_view<int>(), mb.bit_cast_view<int>());

       for (int i = 0; i < Size; i += VL) {
         simd<int, VL> output = vc.select<VL, 1>(i);
         output.copy_to(C + i);
       }
     });
   }).wait();

  int err_cnt = 0;
  for (unsigned i = 0; i < Size && err_cnt < 10; ++i)
    if (C[i] != 1) {
      err_cnt++;
      std::cerr << "Failed at index " << i << ", " << C[i] << " != 1\n";
    }

  free(C, q);
  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
