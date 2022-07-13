// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// XFAIL: hip_nvidia
// SYCL in order and default queue property trace test
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q1;
  sycl::queue q2{sycl::property::queue::in_order()};
  return 0;
}

// Test that out-of-order property is passed to piQueueCreate by default
// CHECK: ---> piQueueCreate(
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : 1

// Test that out-of-order property is not passed to piQueueCreate when
// property::queue::in_order() is passed to the SYCL queue
// CHECK: ---> piQueueCreate(
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : {{0[xX]?[0-9a-fA-F]*}}
// CHECK-NEXT:  <unknown> : 0
