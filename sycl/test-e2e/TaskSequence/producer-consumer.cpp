//==-------------- producer-consumer.cpp - DPC++ task_sequence -------------==//
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

constexpr int kSize = 128;

using intertask_pipe = ext::intel::pipe<class p, int>;

template <typename OutPipe> void producer(int *a, int *b, int N) {
  for (int i = 0; i < N; ++i) {
    OutPipe::write(a[i] * b[i]);
  }
}

template <typename InPipe> int consumer(int N) {
  int total = 0;
  for (int i = 0; i < N; ++i) {
    total += (InPipe::read() + 42);
  }
  return total;
}

int main() {
  queue myQueue;

  int result = 0;
  buffer<int, 1> res_buf(&result, range<1>(1));

  myQueue.submit([&](handler &cgh) {
    auto res_acc = res_buf.get_access<access::mode::write>(cgh);
    cgh.single_task([=](kernel_handler kh) {
      int d1[kSize], d2[kSize];
      for (int i = 0; i < kSize; ++i)
        d1[i] = d2[i] = i;
      task_sequence<producer<intertask_pipe>> producerTask;
      task_sequence<consumer<intertask_pipe>> consumerTask;

      producerTask.async(d1, d2, kSize);
      consumerTask.async(kSize);
      res_acc[0] = consumerTask.get();
    });
  });
  myQueue.wait();

  // Check result:
  int sum = 0;
  for (int i = 0; i < kSize; ++i)
    sum += i * i + 42;
  assert(result == sum);
  return 0;
}