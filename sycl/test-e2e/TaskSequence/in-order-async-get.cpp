//==------------- in-order-async-get.cpp - DPC++ task_sequence -------------==//
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

constexpr int kNumInputs = 16;

class InOrderTasks;

// task that computes the sigmoid function
double sigmoid(double in) {
  double Ex = sycl::exp(in);
  return (Ex / (Ex + 1));
}

int main() {
  std::vector<double> values(kNumInputs);
  std::vector<double> golden(kNumInputs);
  std::vector<double> results(kNumInputs, 0);

  // test data
  for (int i = 0; i < kNumInputs; ++i) {
    values[i] = (static_cast<double>(-kNumInputs / 2.0 + i) / 1000);
    golden[i] = 1 / (1 + exp(-values[i]));
  }

  {
    queue q;

    buffer values_buf(values);
    buffer results_buf(values);

    q.submit([&](handler &h) {
      accessor values_acc(values_buf, h, read_only);
      accessor results_acc(results_buf, h, write_only);
      h.single_task<InOrderTasks>([=]() {
        task_sequence<sigmoid,
                      decltype(properties{invocation_capacity<kNumInputs>})>
            ts;

        for (int i = 0; i < kNumInputs; ++i) {
          ts.async(values_acc[i]);
        }

        // multiple async calls before first get call
        for (int i = 0; i < kNumInputs; ++i) {
          results_acc[i] = ts.get();
        }
      });
    });
    q.wait();
  }

  // verification
  for (int i = 0; i < kNumInputs; ++i) {
    assert(abs(results[i] - golden[i]) < 0.001);
  }

  return 0;
}
