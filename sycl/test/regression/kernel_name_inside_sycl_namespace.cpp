// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_DISABLE_NAMESPACE_INLINE__ %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

//==------- kernel_name_inside_sycl_namespace.cpp - Regression test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

struct A {};
namespace sycl {
struct B {};
using b_t = B;
using a_t = A;
} // namespace sycl

int main() {
  cl::sycl::queue Queue;
  Queue.submit(
      [&](cl::sycl::handler &CGH) { CGH.single_task<sycl::b_t>([=]() {}); });
  Queue.submit(
      [&](cl::sycl::handler &CGH) { CGH.single_task<sycl::a_t>([=]() {}); });
  return 0;
}
