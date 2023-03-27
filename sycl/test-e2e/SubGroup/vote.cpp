// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--------------- vote.cpp - SYCL sub_group vote test --*- C++ -*---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

void check(queue Queue, const int G, const int L, const int D, const int R) {
  try {
    range<1> GRange(G), LRange(L);
    nd_range<1> NdRange(GRange, LRange);
    buffer<int, 1> sganybuf(G);
    buffer<int, 1> sgallbuf(G);

    // Initialise buffer with zeros
    Queue.submit([&](handler &cgh) {
      auto sganyacc = sganybuf.get_access<access::mode::read_write>(cgh);
      auto sgallacc = sgallbuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class init>(range<1>{(unsigned)G}, [=](id<1> index) {
        sganyacc[index] = 0;
        sgallacc[index] = 0;
      });
    });

    Queue.submit([&](handler &cgh) {
      auto sganyacc = sganybuf.get_access<access::mode::read_write>(cgh);
      auto sgallacc = sgallbuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class init_bufs>(NdRange, [=](nd_item<1> NdItem) {
        sganyacc[NdItem.get_global_id()] = 0;
        sgallacc[NdItem.get_global_id()] = 0;
      });
    });

    Queue.submit([&](handler &cgh) {
      auto sganyacc = sganybuf.get_access<access::mode::read_write>(cgh);
      auto sgallacc = sgallbuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class subgr>(NdRange, [=](nd_item<1> NdItem) {
        ext::oneapi::sub_group SG = NdItem.get_sub_group();
        /* Set to 1 if any local ID in subgroup devided by D has remainder R */
        if (ext::oneapi::any_of(SG, SG.get_local_id().get(0) % D == R)) {
          sganyacc[NdItem.get_global_id()] = 1;
        }
        /* Set to 1 if remainder of division of subgroup local ID by D is less
         * than R for all work items in subgroup */
        if (ext::oneapi::all_of(SG, SG.get_local_id().get(0) % D < R)) {
          sgallacc[NdItem.get_global_id()] = 1;
        }
      });
    });
    auto sganyacc = sganybuf.get_access<access::mode::read_write>();
    auto sgallacc = sgallbuf.get_access<access::mode::read_write>();
    for (int j = 0; j < G; j++) {
      exit_if_not_equal(sganyacc[j], (int)(D > R), "any");
      exit_if_not_equal(sgallacc[j], (int)(D <= R), "all");
    }

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check(Queue, 240, 80, 3, 1);
  check(Queue, 24, 12, 3, 4);
  check(Queue, 1024, 256, 3, 1);
  std::cout << "Test passed." << std::endl;
}
