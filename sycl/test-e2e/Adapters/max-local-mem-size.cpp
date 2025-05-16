// REQUIRES: cuda || hip

// DEFINE: %{env_var} = %if cuda %{UR_CUDA_MAX_LOCAL_MEM_SIZE %} %else %{UR_HIP_MAX_LOCAL_MEM_SIZE%}

// RUN: %{build} -o %t.out
// RUN: %{run} %{env_var}=0 %t.out 2>&1 | FileCheck --check-prefixes=CHECK-ZERO %s
// RUN: %{run} %{env_var}=100000000 %t.out 2>&1 | FileCheck --check-prefixes=CHECK-OVERALLOCATE %s

//==------------------------ max-local-mem-size.cpp -----------------------===//
//==--- SYCL test to test UR_{CUDA,HIP}_MAX_LOCAL_MEM_SIZE env var---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>

int main() {
  try {
    sycl::queue Q{};
    auto LocalSize =
        Q.get_device().get_info<sycl::info::device::local_mem_size>();
    Q.submit([&](sycl::handler &cgh) {
       auto LocalAcc = sycl::local_accessor<char>(LocalSize + 1, cgh);
       cgh.parallel_for(sycl::nd_range<1>{32, 32}, [=](sycl::nd_item<1> idx) {
         LocalAcc[idx.get_global_linear_id()] *= 2;
       });
     }).wait();
  } catch (const std::exception &e) {
    std::puts(e.what());
  }
  // CHECK-ZERO: Invalid value specified for
  // CHECK-OVERALLOCATE: Excessive allocation of local memory on the device
}
