// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
//==-------------- common.cpp - SYCL sub_group common test -----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <sycl/sycl.hpp>
using namespace cl::sycl;
struct Data {
  unsigned int local_id;
  unsigned int local_range;
  unsigned int max_local_range;
  unsigned int group_id;
  unsigned int group_range;
};

void check(queue &Queue, unsigned int G, unsigned int L) {

  try {
    nd_range<1> NdRange(G, L);
    buffer<struct Data, 1> syclbuf(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      auto syclacc = syclbuf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class sycl_subgr>(NdRange, [=](nd_item<1> NdItem) {
        ext::oneapi::sub_group SG = NdItem.get_sub_group();
        syclacc[NdItem.get_global_id()].local_id = SG.get_local_id().get(0);
        syclacc[NdItem.get_global_id()].local_range =
            SG.get_local_range().get(0);
        syclacc[NdItem.get_global_id()].max_local_range =
            SG.get_max_local_range().get(0);
        syclacc[NdItem.get_global_id()].group_id = SG.get_group_id().get(0);
        syclacc[NdItem.get_global_id()].group_range =
            SG.get_group_range().get(0);
        if (NdItem.get_global_id(0) == 0)
          sgsizeacc[0] = SG.get_max_local_range()[0];
      });
    });
    auto syclacc = syclbuf.get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
    unsigned int sg_size = sgsizeacc[0];
    unsigned int num_sg = L / sg_size + (L % sg_size ? 1 : 0);
    for (int j = 0; j < G; j++) {
      unsigned int group_id = j % L / sg_size;
      unsigned int local_range =
          (group_id + 1 == num_sg) ? (L - group_id * sg_size) : sg_size;
      exit_if_not_equal(syclacc[j].local_id, j % L % sg_size, "local_id");
      exit_if_not_equal(syclacc[j].local_range, local_range, "local_range");
      exit_if_not_equal(syclacc[j].max_local_range, syclacc[0].max_local_range,
                        "max_local_range");
      exit_if_not_equal(syclacc[j].group_id, group_id, "group_id");
      exit_if_not_equal(syclacc[j].group_range, num_sg, "group_range");
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}
int main() {
  queue Queue;
  if (Queue.get_device().is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

  check(Queue, 240, 80);
  check(Queue, 8, 4);
  check(Queue, 24, 12);
  check(Queue, 1024, 256);
  std::cout << "Test passed." << std::endl;
  return 0;
}
