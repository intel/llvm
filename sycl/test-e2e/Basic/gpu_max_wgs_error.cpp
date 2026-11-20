//===------------------------gpu_max_wgs_error.cpp-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies that submitting a kernel with an oversized work-group throws
/// sycl::errc::nd_range with the expected backend-specific diagnostic.
///
//===----------------------------------------------------------------------===//
// REQUIRES: gpu

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22300

// RUN: %{build} -DNVIDIA -o %t.nvidia.out -fno-sycl-id-queries-fit-in-int
// RUN: %{build} -o %t.out -fno-sycl-id-queries-fit-in-int
// RUN: %{run} %if cuda %{ %t.nvidia.out %} %else %{ %t.out %}

#include <cassert>
#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>

using namespace sycl;

const std::string expected_msg =
#ifdef NVIDIA
    "Exceeded the number of registers available on the hardware";
#else
    "Total number of work-items in a work-group cannot exceed";
#endif

template <int N> void check(queue &q, range<N> global, range<N> local) {
  try {
    q.parallel_for(nd_range<N>(global, local), [=](nd_item<N> item) {}).wait();
    assert(false && "Exception expected");
  } catch (const sycl::exception &e) {
    std::string msg = e.what();
    assert(e.code() == errc::nd_range && "Unexpected error code");
    assert(msg.rfind(expected_msg, 0) == 0 && "Unexpected error message");
  }
}

int main() {
  queue q;
  device d = q.get_device();

  range<2> max_2 = d.get_info<sycl::info::device::max_work_item_sizes<2>>();
  check(q, max_2, max_2);

  range<3> max_3 = d.get_info<sycl::info::device::max_work_item_sizes<3>>();
  check(q, max_3, max_3);

  return 0;
}
