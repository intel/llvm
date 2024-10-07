// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==----------------- in_order_barrier_profiling.cpp -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Level Zero adapter has a similar in-order queue barrier optimization that
// leads to incorrect profiling values.
// https://github.com/intel/llvm/issues/14186
// UNSUPPORTED: level_zero || opencl
#include <sycl/detail/core.hpp>

#include <sycl/properties/all_properties.hpp>

using namespace sycl;

// Checks that the barrier profiling info is consistent with the previous
// command, despite the fact that the latter started after the barrier was
// submitted.
int main() {
  queue Q({property::queue::in_order(), property::queue::enable_profiling()});

  buffer<int, 1> Buf(range<1>(1));
  event KernelEvent;
  event BarrierEvent;
  {
    auto HostAcc = Buf.get_access();
    KernelEvent = Q.submit([&](handler &cgh) {
      auto Acc = Buf.get_access(cgh);
      cgh.single_task([=]() {});
    });
    BarrierEvent = Q.ext_oneapi_submit_barrier();
  }
  uint64_t KernelEnd =
      KernelEvent.get_profiling_info<info::event_profiling::command_end>();
  uint64_t BarrierStart =
      BarrierEvent.get_profiling_info<info::event_profiling::command_start>();
  assert(KernelEnd <= BarrierStart);
}
