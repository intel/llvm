// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------- event_dependency.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A cross-context dependency stated explicitly with handler::depends_on(),
// rather than derived from a buffer requirement. The runtime handles the two
// separately: an explicit event dependency carries no requirement, so there is
// no memory transfer to hang the ordering on - the consuming command's wait
// list is all there is.
//
// The producing command's data is still read through a buffer, so that a
// consumer which fails to wait is caught by the values it reads.

#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

constexpr size_t N = 256;
constexpr int Repeats = 4096;

int main() {
  sycl::device Dev;

  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev};
  sycl::queue Q2{Ctx2, Dev};

  std::vector<int> Produced(N, 0);
  std::vector<int> Consumed(N, 0);
  {
    sycl::buffer<int, 1> ProducedBuf(Produced.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> ConsumedBuf(Consumed.data(), sycl::range<1>(N));

    sycl::event E1 = Q1.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{ProducedBuf, CGH, sycl::write_only, sycl::no_init};
      CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
        int Value = 0;
        for (int It = 0; It < Repeats; ++It)
          Value += static_cast<int>(I) + 1;
        Acc[I] = Value / Repeats;
      });
    });

    // E1 belongs to Ctx1, the command below to Ctx2.
    sycl::event E2 = Q2.submit([&](sycl::handler &CGH) {
      CGH.depends_on(E1);
      sycl::accessor In{ProducedBuf, CGH, sycl::read_only};
      sycl::accessor Out{ConsumedBuf, CGH, sycl::write_only, sycl::no_init};
      CGH.parallel_for(sycl::range<1>(N),
                       [=](sycl::id<1> I) { Out[I] = In[I] * 2; });
    });

    // Waiting on the consuming event alone has to be enough: it transitively
    // depends on the producing one.
    E2.wait_and_throw();
  }

  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    const int Expected = (static_cast<int>(I) + 1) * 2;
    if (Consumed[I] != Expected) {
      std::cout << "Consumed[" << I << "] == " << Consumed[I] << ", expected "
                << Expected << std::endl;
      ++Failures;
    }
  }

  std::cout << (Failures == 0 ? "Test passed" : "Test failed") << std::endl;
  return Failures == 0 ? 0 : 1;
}
