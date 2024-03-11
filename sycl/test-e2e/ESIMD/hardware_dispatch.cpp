//==---------------- hardware_dispatch.cpp  - ESIMD hardware dispatch test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_bdw %s -o %t.out
// RUN: %t.out
// TODO: remove XFAIL when the fix in GPU RT for Windows is updated on CI
// machines
// XFAIL: windows
// This is basic test to test hardware dispatch functionality with ESIMD.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/sycl.hpp>
#include <vector>

using shared_allocator = sycl::usm_allocator<int32_t, sycl::usm::alloc::shared>;
using shared_vector = std::vector<int32_t, shared_allocator>;
using namespace sycl::ext::intel::esimd;

int main() {
  sycl::queue q;
  shared_allocator allocator(q);

  shared_vector vec(4, allocator);
  {
    q.submit([&](sycl::handler &cgh) {
      int *output_ptr = vec.data();
      cgh.single_task<class test1>([=]() SYCL_ESIMD_KERNEL {
        simd<int, 4> result;

        // test if_architecture_is
        sycl::ext::oneapi::experimental::if_architecture_is<
            sycl::ext::oneapi::experimental::architecture::intel_gpu_bdw>(
            [&]() { result[0] = 1; })
            .otherwise([&]() { result[0] = 0; });

        // test else_if_architecture_is
        sycl::ext::oneapi::experimental::if_architecture_is<
            sycl::ext::oneapi::experimental::architecture::intel_gpu_dg1>(
            [&]() { result[1] = 0; })
            .else_if_architecture_is<
                sycl::ext::oneapi::experimental::architecture::intel_gpu_bdw>(
                [&]() { result[1] = 2; })
            .otherwise([&]() { result[1] = 0; });

        // test otherwise
        sycl::ext::oneapi::experimental::if_architecture_is<
            sycl::ext::oneapi::experimental::architecture::intel_gpu_dg1>(
            [&]() { result[2] = 0; })
            .otherwise([&]() { result[2] = 3; });

        // test more than one architecture template parameter is passed to
        // if_architecture_is
        sycl::ext::oneapi::experimental::if_architecture_is<
            sycl::ext::oneapi::experimental::architecture::intel_gpu_dg1,
            sycl::ext::oneapi::experimental::architecture::intel_gpu_bdw>(
            [&]() { result[3] = 4; })
            .otherwise([&]() { result[3] = 0; });
        result.copy_to(output_ptr);
      });
    });
  }
  q.wait();

  for (int i = 0; i < 4; ++i) {
    if (vec[i] != i + 1) {
      std::cout << "Test failed for index " << i << std::endl;
      return 1;
    }
  }
  std::cout << "Pass" << std::endl;
  return 0;
}
