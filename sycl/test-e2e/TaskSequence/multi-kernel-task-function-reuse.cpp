//==------ multi-kernel-task-function-reuse.cpp - DPC++ task_sequence ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME: failure in post-commit, re-enable when fixed:
// UNSUPPORTED: linux

// REQUIRES: aspect-ext_intel_fpga_task_sequence
// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

constexpr int kCount = 1024;

int vectorSum(const int *v, size_t s, size_t sz) {
  int result = 0;
  for (size_t i = s; i < s + sz; i++) {
    result += v[i];
  }

  return result;
}

// Kernel identifiers
class SequentialTask;
class ParallelTask;

int main() {
  queue q;

  // create input and golden output data
  std::vector<int> in(kCount), out(2);
  for (int i = 0; i < kCount; i++) {
    in[i] = i;
  }

  int golden = vectorSum(in.data(), 0, kCount);

  {
    buffer in_buf(in);
    buffer out_buf(out);

    q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(out_buf, h, write_only);
      h.single_task<SequentialTask>([=]() {
        task_sequence<vectorSum> whole;
        whole.async(in_acc.get_pointer(), 0, kCount);
        out_acc[0] = whole.get();
      });
    });

    q.submit([&](sycl::handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(out_buf, h, write_only);
      h.single_task<ParallelTask>([=]() {
        task_sequence<vectorSum> firstQuarter;
        task_sequence<vectorSum> secondQuarter;
        task_sequence<vectorSum> thirdQuarter;
        task_sequence<vectorSum> fourthQuarter;
        constexpr int quarterCount = kCount / 4;
        firstQuarter.async(in_acc.get_pointer(), 0, quarterCount);
        secondQuarter.async(in_acc.get_pointer(), quarterCount, quarterCount);
        thirdQuarter.async(in_acc.get_pointer(), 2 * quarterCount,
                           quarterCount);
        fourthQuarter.async(in_acc.get_pointer(), 3 * quarterCount,
                            quarterCount);
        out_acc[1] = firstQuarter.get() + secondQuarter.get() +
                     thirdQuarter.get() + fourthQuarter.get();
      });
    });
    q.wait();
  }

  assert(out[0] == golden);
  assert(out[1] == golden);
  return 0;
}
