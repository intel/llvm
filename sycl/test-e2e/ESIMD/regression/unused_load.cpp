//==---------- unused_load.cpp  - DPC++ ESIMD on-device test ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: gpu-intel-gen9 && windows
// UNSUPPORTED: cuda || hip
// RUN: %{build} -I%S/.. -o %t.out
// RUN: %{run} %t.out

// This test checks that ESIMD JIT compilation does not crash on unused
// copy_from invocation.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

constexpr unsigned int VL = 16;

using Ty = float;

int main() {
  Ty data0[VL] = {0};

  try {
    queue q;

    buffer<Ty, 1> buf0(data0, range<1>(VL));

    q.submit([&](handler &cgh) {
      std::cout << "Running on "
                << q.get_device().get_info<sycl::info::device::name>() << "\n";

      auto acc0 = buf0.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class Test>(range<1>(1),
                                   [=](sycl::id<1> i) SYCL_ESIMD_KERNEL {
                                     using namespace sycl::ext::intel::esimd;
                                     simd<Ty, VL> var;
                                     var.copy_from(acc0, 0);
                                   });
    });
    q.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Passed\n";
  return 0;
}
