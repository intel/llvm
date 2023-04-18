// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//=-queue_parallel_for_generic.cpp - SYCL queue parallel_for generic lambda-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>

int main() {
  sycl::queue q{sycl::property::queue::in_order()};
  auto dev = q.get_device();
  auto ctx = q.get_context();
  constexpr int N = 8;

  if (!dev.get_info<sycl::info::device::usm_shared_allocations>()) {
    return 0;
  }

  auto A = static_cast<int *>(sycl::malloc_shared(N * sizeof(int), dev, ctx));

  for (int i = 0; i < N; i++) {
    A[i] = 1;
  }

  q.parallel_for<class Bar>(N, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
    A[i]++;
  });

  q.parallel_for<class Foo>({N}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
    A[i]++;
  });

  sycl::id<1> offset(0);
  q.parallel_for<class Baz>(sycl::range<1>{N}, offset, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
    A[i]++;
  });

  sycl::nd_range<1> NDR(sycl::range<1>{N}, sycl::range<1>{2});
  q.parallel_for<class NDFoo>(NDR, [=](auto nd_i) {
    static_assert(std::is_same<decltype(nd_i), sycl::nd_item<1>>::value,
                  "lambda arg type is unexpected");
    auto i = nd_i.get_global_id(0);
    A[i]++;
  });

  q.wait();

  for (int i = 0; i < N; i++) {
    assert(A[i] == 5);
  }
  sycl::free(A, ctx);
}
