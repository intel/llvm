// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
//==--------------- linear-sub_group.cpp - SYCL linear id test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../SubGroup/helper.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main(int argc, char *argv[]) {
  queue q;
  if (!core_sg_supported(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  // Fill output array with sub-group IDs
  const uint32_t outer = 2;
  const uint32_t inner = 8;
  std::vector<int> output(outer * inner, 0);
  {
    buffer<int, 2> output_buf(output.data(), range<2>(outer, inner));
    q.submit([&](handler &cgh) {
      auto output = output_buf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class linear_id>(
          nd_range<2>(range<2>(outer, inner), range<2>(outer, inner)),
          [=](nd_item<2> it) {
            id<2> idx = it.get_global_id();
            ext::oneapi::sub_group sg = it.get_sub_group();
            output[idx] = sg.get_group_id()[0] * sg.get_local_range()[0] +
                          sg.get_local_id()[0];
          });
    });
  }

  // Compare with expected result
  for (int idx = 0; idx < outer * inner; ++idx) {
    assert(output[idx] == idx);
  }
  std::cout << "Test passed.\n";
  return 0;
}
