//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-unnamed-lambda %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// SYCL ordered queue kernel shortcut test
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  queue q{property::queue::in_order()};
  auto dev = q.get_device();
  auto ctx = q.get_context();
  const int N = 8;
  int err_cnt = 0;

  if (dev.get_info<info::device::usm_shared_allocations>()) {
    auto A = (int *)malloc_shared(N * sizeof(int), dev, ctx);

    for (int i = 0; i < N; i++) {
      A[i] = 1;
    }

    q.parallel_for(range<1>{N}, [=](id<1> ID) {
      auto i = ID[0];
      A[i]++;
    });

    q.parallel_for<class Foo>(range<1>{N}, [=](id<1> ID) {
      auto i = ID[0];
      A[i]++;
    });

    q.single_task([=]() {
      for (int i = 0; i < N; i++) {
        A[i]++;
      }
    });

    q.single_task<class Bar>([=]() {
      for (int i = 0; i < N; i++) {
        A[i]++;
      }
    });

    id<1> offset(0);
    q.parallel_for(range<1>{N}, offset, [=](id<1> ID) {
      auto i = ID[0];
      A[i]++;
    });

    q.parallel_for<class Baz>(range<1>{N}, offset, [=](id<1> ID) {
      auto i = ID[0];
      A[i]++;
    });

    nd_range<1> NDR(range<1>{N}, range<1>{2});
    q.parallel_for(NDR, [=](nd_item<1> Item) {
      auto i = Item.get_global_id(0);
      A[i]++;
    });

    q.parallel_for<class NDFoo>(NDR, [=](nd_item<1> Item) {
      auto i = Item.get_global_id(0);
      A[i]++;
    });

    q.submit([&](handler &cgh) {
      cgh.parallel_for_work_group<class WkGrp>(
          range<1>{N / 2}, range<1>{2}, [=](group<1> myGroup) {
            auto j = myGroup.get_id(0);
            myGroup.parallel_for_work_item(
                [&](h_item<1> it) { A[(j * 2) + it.get_local_id(0)]++; });
          });
    });

    q.submit([&](handler &cgh) {
      cgh.parallel_for_work_group(
          range<1>{N / 2}, range<1>{2}, [=](group<1> myGroup) {
            auto j = myGroup.get_id(0);
            myGroup.parallel_for_work_item(
                [&](h_item<1> it) { A[(j * 2) + it.get_local_id(0)]++; });
          });
    });

    q.wait();

    for (int i = 0; i < N; i++) {
      if (A[i] != 11) {
        std::cerr << "Mismatch at index " << i << " : " << A[i]
                  << " != 11 (expected)" << std::endl;
        err_cnt++;
      }
    }
    free(A, ctx);
    if (err_cnt != 0) {
      std::cerr << "Total mismatch =  " << err_cnt << std::endl;
      return 1;
    }
  }

  std::cout << "Passed\n";
  return 0;
}
