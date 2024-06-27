// UNSUPPORTED: accelerator
// TODO: FPGAs currently report `sub_group_sizes` as non-empty list,
// despite not having extension `cl_intel_required_subgroup_size`
// UNSUPPORTED: cuda || hip
// TODO: Similar issue to FPGAs

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==-- get_subgroup_sizes.cpp - Test for bug fix in subgroup sizes query --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue Q;
  auto Dev = Q.get_device();
  auto Vec = Dev.get_info<info::device::extensions>();
  std::vector<size_t> SubGroupSizes =
      Dev.get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(Vec.begin(), Vec.end(), "cl_intel_required_subgroup_size") !=
      std::end(Vec)) {
    assert(!SubGroupSizes.empty() &&
           "Required sub-group size list should not be empty");
  } else {
    assert(SubGroupSizes.empty() &&
           "Required sub-group size list should be empty");
  }
  return 0;
}
