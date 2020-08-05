// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==-- get_subgroup_sizes.cpp - Test for bug fix in subgroup sizes query --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  queue Q;
  auto Dev = Q.get_device();
  if (Dev.has_extension("cl_intel_required_subgroup_size")) {
    cl::sycl::vector_class<size_t> SubGroupSizes =
        Dev.get_info<cl::sycl::info::device::sub_group_sizes>();
    cl::sycl::vector_class<size_t>::const_iterator MaxIter =
        std::max_element(SubGroupSizes.begin(), SubGroupSizes.end());
    int MaxSubGroup_size = *MaxIter;
  }
  return 0;
}
