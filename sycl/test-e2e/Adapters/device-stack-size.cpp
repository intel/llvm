// REQUIRES: cuda || hip

// RUN: %{build} -o %t.out
// RUN: %{run} %if cuda %{UR_CUDA_STACK_SIZE%} %else %{UR_HIP_STACK_SIZE%}=0 %t.out 2>&1 | FileCheck --check-prefixes=CHECK-INVALID %s
// RUN: %{run} %if cuda %{UR_CUDA_STACK_SIZE%} %else %{UR_HIP_STACK_SIZE%}=abc %t.out 2>&1 | FileCheck --check-prefixes=CHECK-INVALID %s
// RUN: %{run} %if cuda %{UR_CUDA_STACK_SIZE%} %else %{UR_HIP_STACK_SIZE%}=16384 %t.out 2>&1 | FileCheck --check-prefixes=CHECK-VALID %s

//==-------------------------- device-stack-size.cpp ----------------------===//
//==--- SYCL test to test UR_{CUDA,HIP}_STACK_SIZE env var ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifies the behavior of the per-thread stack size limit env var:
//   * An invalid value (non-positive or non-numeric) is rejected with a
//     diagnostic at device initialization.
//   * A valid value is accepted and device initialization plus a trivial
//     kernel launch succeed.

#include <cstdio>
#include <sycl/detail/core.hpp>

int main() {
  try {
    sycl::queue Q{};
    Q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {});
     }).wait();
    std::puts("PASS");
  } catch (const std::exception &e) {
    std::puts(e.what());
  }
  // CHECK-INVALID: Invalid value specified for
  // CHECK-VALID: PASS
}
