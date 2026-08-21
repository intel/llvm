// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------- buffer_dependency.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A dependency between two commands of different contexts cannot be expressed
// in the backend. The runtime creates a host-signalled proxy event in the
// consuming command's context and signals it once the producing event has
// retired, or - where the backend cannot do that - connects the two contexts
// with an empty host task.
//
// Here the dependency comes from a buffer used in both contexts: the consuming
// command must not run before the producing one, or it reads values that have
// not been written yet.

#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

constexpr size_t N = 256;
// Enough work for a consumer that ignores the dependency to be caught reading
// the buffer too early.
constexpr int Repeats = 4096;

int main() {
  sycl::device Dev;

  // Two explicitly created contexts, so that the queues below do not share the
  // platform's default context.
  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev};
  sycl::queue Q2{Ctx2, Dev};

  std::vector<int> Data(N, 0);
  {
    sycl::buffer<int, 1> Buf(Data.data(), sycl::range<1>(N));

    // Produced in Ctx1.
    Q1.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::write_only, sycl::no_init};
      CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
        int Value = 0;
        for (int It = 0; It < Repeats; ++It)
          Value += static_cast<int>(I) + 1;
        Acc[I] = Value / Repeats;
      });
    });

    // Consumed in Ctx2, depending on the command above through the buffer.
    Q2.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::read_write};
      CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) { Acc[I] *= 2; });
    });

    Q1.wait_and_throw();
    Q2.wait_and_throw();
  }

  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    const int Expected = (static_cast<int>(I) + 1) * 2;
    if (Data[I] != Expected) {
      std::cout << "Data[" << I << "] == " << Data[I] << ", expected "
                << Expected << std::endl;
      ++Failures;
    }
  }

  std::cout << (Failures == 0 ? "Test passed" : "Test failed") << std::endl;
  return Failures == 0 ? 0 : 1;
}
