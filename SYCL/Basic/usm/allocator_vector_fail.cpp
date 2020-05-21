// XFAIL: cuda
// piextUSM*Alloc functions for CUDA are not behaving as described in
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/cl_intel_unified_shared_memory.asciidoc
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: %HOST_RUN_PLACEHOLDER %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==-- allocator_vector_fail.cpp - Device Memory Allocator fail test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <vector>

using namespace cl::sycl;

const int N = 8;

class foo;
int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (dev.get_info<info::device::usm_device_allocations>()) {
    try {
      usm_allocator<int, usm::alloc::device> alloc(ctxt, dev);
      std::vector<int, decltype(alloc)> vec(alloc);

      // This statement should throw an exception since
      // device pointers may not be accessed on the host.
      vec.assign(N, 42);
    } catch (feature_not_supported) {
      return 0;
    }

    return -1;
  }
  return 0;
}
