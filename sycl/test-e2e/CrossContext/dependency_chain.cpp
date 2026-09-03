// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------- dependency_chain.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A long chain of commands alternating between two contexts, so that every step
// depends on a command of the other context. Each step needs its own connection
// - a proxy event of its own where the backend supports one - and the result is
// only correct if all of them are ordered.
//
// This also covers the housekeeping around the mechanism: many proxies are
// created and signalled over the run of the program, and none of them may be
// left behind.

#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

constexpr size_t N = 128;
constexpr int Steps = 64;

int main() {
  sycl::device Dev;

  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev};
  sycl::queue Q2{Ctx2, Dev};

  std::vector<int> Data(N, 0);
  {
    sycl::buffer<int, 1> Buf(Data.data(), sycl::range<1>(N));

    sycl::event Prev;
    for (int Step = 0; Step < Steps; ++Step) {
      // Alternating queues, so the previous step is always foreign.
      sycl::queue &Q = (Step % 2 == 0) ? Q1 : Q2;
      Prev = Q.submit([&](sycl::handler &CGH) {
        if (Step > 0)
          CGH.depends_on(Prev);
        sycl::accessor Acc{Buf, CGH, sycl::read_write};
        CGH.parallel_for(sycl::range<1>(N),
                         [=](sycl::id<1> I) { Acc[I] += 1; });
      });
    }
    Prev.wait_and_throw();
  }

  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    if (Data[I] != Steps) {
      std::cout << "Data[" << I << "] == " << Data[I] << ", expected " << Steps
                << std::endl;
      ++Failures;
    }
  }

  std::cout << (Failures == 0 ? "Test passed" : "Test failed") << std::endl;
  return Failures == 0 ? 0 : 1;
}
