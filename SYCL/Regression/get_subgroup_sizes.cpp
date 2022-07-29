// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==-- get_subgroup_sizes.cpp - Test for bug fix in subgroup sizes query --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue Q;
  auto Dev = Q.get_device();
  auto Vec = Dev.get_info<info::device::extensions>();
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_required_subgroup_size") !=
      std::end(Vec)) {
    std::vector<size_t> SubGroupSizes =
        Dev.get_info<sycl::info::device::sub_group_sizes>();
    std::vector<size_t>::const_iterator MaxIter =
        std::max_element(SubGroupSizes.begin(), SubGroupSizes.end());
    int MaxSubGroup_size = *MaxIter;
  }
  return 0;
}
