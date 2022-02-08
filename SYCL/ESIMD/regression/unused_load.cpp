//==---------- unused_load.cpp  - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// TODO: esimd_emulator fails due to unimplemented __esimd_oword_ld_unaligned
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks that ESIMD JIT compilation does not crash on unused
// copy_from invocation.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

#include <iostream>

using namespace cl::sycl;

constexpr unsigned int VL = 16;

using Ty = float;

int main() {
  Ty data0[VL] = {0};

  try {
    queue q;

    buffer<Ty, 1> buf0(data0, range<1>(VL));

    q.submit([&](handler &cgh) {
      std::cout << "Running on "
                << q.get_device().get_info<cl::sycl::info::device::name>()
                << "\n";

      auto acc0 = buf0.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class Test>(
          range<1>(1), [=](sycl::id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::experimental::esimd;
            simd<Ty, VL> var;
            var.copy_from(acc0, 0);
          });
    });
    q.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Passed\n";
  return 0;
}
