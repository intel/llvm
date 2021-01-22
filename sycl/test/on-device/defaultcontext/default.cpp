// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==----------- default.cpp - SYCL default context test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::queue q2;

  auto c1 = q.get_context();
  auto c2 = q2.get_context();

  auto d1 = q.get_device();
  auto d2 = q2.get_device();

  assert(d1 == d2);
  assert(c1 == c2);

  return 0;
}
