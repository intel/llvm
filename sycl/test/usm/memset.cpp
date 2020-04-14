// XFAIL: cuda
// piextUSM*Alloc functions for CUDA are not behaving as described in
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/cl_intel_unified_shared_memory.asciidoc
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==---- memset.cpp - USM memset test --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

using namespace cl::sycl;

static constexpr int count = 100;

int main() {
  queue q([](exception_list el) {
    for (auto &e : el)
      std::rethrow_exception(e);
  });
  if (q.get_device().get_info<info::device::usm_shared_allocations>()) {
    uint32_t *src = (uint32_t *)malloc_shared(sizeof(uint32_t) * count,
                                              q.get_device(), q.get_context());

    event init_copy = q.submit(
        [&](handler &cgh) { cgh.memset(src, 0x15, sizeof(uint32_t) * count); });

    q.submit([&](handler &cgh) {
      cgh.depends_on(init_copy);
      cgh.single_task<class double_dest>([=]() {
        for (int i = 0; i < count; i++)
          src[i] *= 2;
      });
    });
    q.wait_and_throw();

    for (int i = 0; i < count; i++) {
      assert(src[i] == 0x2a2a2a2a);
    }

    try {
      // Filling to nullptr should throw.
      q.submit([&](handler &cgh) {
        cgh.memset(nullptr, 0, sizeof(uint32_t) * count);
      });
      q.wait_and_throw();
      assert(false && "Expected error from writing to nullptr");
    } catch (runtime_error e) {
    }
  }
  return 0;
}
