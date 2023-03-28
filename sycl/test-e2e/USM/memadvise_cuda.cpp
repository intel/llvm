// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t1.out
// REQUIRES: cuda
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out
// RUN: %ACC_RUN_PLACEHOLDER %t1.out

//==---------------- memadvise_cuda.cpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

int main() {
  const size_t size = 100;
  queue q;
  auto dev = q.get_device();
  auto ctx = q.get_context();
  if (!dev.get_info<info::device::usm_shared_allocations>()) {
    std::cout << "Shared USM is not supported. Skipping test." << std::endl;
    return 0;
  }

  void *ptr = malloc_shared(size, dev, ctx);
  if (ptr == nullptr) {
    std::cout << "Allocation failed!" << std::endl;
    return -1;
  }

  std::vector<int> valid_advices{
      PI_MEM_ADVICE_CUDA_SET_READ_MOSTLY,
      PI_MEM_ADVICE_CUDA_UNSET_READ_MOSTLY,
      PI_MEM_ADVICE_CUDA_SET_PREFERRED_LOCATION,
      PI_MEM_ADVICE_CUDA_UNSET_PREFERRED_LOCATION,
      PI_MEM_ADVICE_CUDA_SET_ACCESSED_BY,
      PI_MEM_ADVICE_CUDA_UNSET_ACCESSED_BY,
      PI_MEM_ADVICE_CUDA_SET_PREFERRED_LOCATION_HOST,
      PI_MEM_ADVICE_CUDA_UNSET_PREFERRED_LOCATION_HOST,
      PI_MEM_ADVICE_CUDA_SET_ACCESSED_BY_HOST,
      PI_MEM_ADVICE_CUDA_UNSET_ACCESSED_BY_HOST,
  };
  for (int advice : valid_advices) {
    q.mem_advise(ptr, size, advice);
  }

  q.wait_and_throw();
  std::cout << "Test passed." << std::endl;
  return 0;
}
