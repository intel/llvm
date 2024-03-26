//==-------------- concurrent-loops.cpp - DPC++ task_sequence --------------==//
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

constexpr int kPrecision = 20;
constexpr double kX = 0.9;
constexpr double kA = 2.5;
constexpr double kB = 3.7;

using intertask_pipe = ext::intel::pipe<class p, double>;

template <typename OutPipe> double task_a(double x, double a, int precision) {
  double res = 1;
  double term = 1;
  double multiplier = a;

  // compute the non-constant common term of the Maclaurin series of exp(x)
  // feed reuseable result through intertask pipe
  // compute remaining terms of Maclaurin series of exp(ax) and accummulate
  for (int iter = 1; iter <= precision; ++iter) {
    term *= x / iter;
    // reuse the shared value for both exponential series
    OutPipe::write(term);
    res += term * multiplier;
    multiplier *= a;
  }

  return res;
}

template <typename InPipe> double task_b(double b, int precision) {
  double res = 1;
  double multiplier = b;

  // reuse the non-constant common term from task_a
  for (int iter = 1; iter <= precision; ++iter) {
    // compute terms of Maclaurin series of exp(bx) and accumulate
    res += InPipe::read() * multiplier;
    multiplier *= b;
  }

  return res;
};

class ConcurrentLoop;

int main() {

  double golden = exp(kA * kX) / exp(kB * kX);

  // initialize inputs
  std::vector<double> in = {kX, kA, kB};
  std::vector<double> out = {0.0};

  {
    sycl::queue q;

    buffer in_buf(in);
    buffer out_buf(out);

    q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(out_buf, h, write_only);
      h.single_task<ConcurrentLoop>([=]() {
        task_sequence<task_a<intertask_pipe>> ts_a;
        task_sequence<task_b<intertask_pipe>> ts_b;

        double x = in_acc[0];
        double a = in_acc[1];
        double b = in_acc[2];

        // approximate e^(ax)/e^(bx) using Maclaurin series
        ts_a.async(x, a, kPrecision);
        ts_b.async(b, kPrecision);

        out_acc[0] = ts_a.get() / ts_b.get();
      });
    });
    q.wait();
  }

  assert(abs(out[0] - golden) < 0.001);
  return 0;
}
