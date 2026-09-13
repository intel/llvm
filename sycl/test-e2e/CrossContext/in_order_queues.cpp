// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------- in_order_queues.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The same cross-context dependencies, but between in-order queues. An in-order
// queue orders its own commands implicitly, and the runtime prunes from a
// command's wait list the dependencies that this order already implies. A
// dependency on a command of another context is not implied by anything and
// must survive that pruning - including when it is represented by a proxy
// event.
//
// The operations below do not commute, so the result is only correct if every
// step ran in submission order.

#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

#include <iostream>
#include <vector>

constexpr size_t N = 256;
constexpr int Rounds = 6;
constexpr int Repeats = 4096;

int main() {
  sycl::device Dev;

  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev, sycl::property::queue::in_order{}};
  sycl::queue Q2{Ctx2, Dev, sycl::property::queue::in_order{}};

  std::vector<int> Data(N, 0);
  {
    sycl::buffer<int, 1> Buf(Data.data(), sycl::range<1>(N));

    for (int Round = 0; Round < Rounds; ++Round) {
      // Two commands per queue, so that each queue has both a local dependency
      // (same queue, implied by the order and pruned) and a foreign one
      // (cross-context, which has to be kept).
      sycl::event A = Q1.submit([&](sycl::handler &CGH) {
        sycl::accessor Acc{Buf, CGH, sycl::read_write};
        CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
          // Busy work, to widen the window for a consumer that fails to wait.
          int One = 0;
          for (int It = 0; It < Repeats; ++It)
            One += 1;
          Acc[I] = Acc[I] * 2 + One / Repeats;
        });
      });
      Q1.submit([&](sycl::handler &CGH) {
        sycl::accessor Acc{Buf, CGH, sycl::read_write};
        CGH.parallel_for(sycl::range<1>(N),
                         [=](sycl::id<1> I) { Acc[I] += 3; });
      });

      sycl::event C = Q2.submit([&](sycl::handler &CGH) {
        CGH.depends_on(A);
        sycl::accessor Acc{Buf, CGH, sycl::read_write};
        CGH.parallel_for(sycl::range<1>(N),
                         [=](sycl::id<1> I) { Acc[I] *= 2; });
      });
      Q2.submit([&](sycl::handler &CGH) {
        CGH.depends_on(C);
        sycl::accessor Acc{Buf, CGH, sycl::read_write};
        CGH.parallel_for(sycl::range<1>(N),
                         [=](sycl::id<1> I) { Acc[I] += 1; });
      });
    }

    Q1.wait_and_throw();
    Q2.wait_and_throw();
  }

  // The same sequence, on the host.
  int Expected = 0;
  for (int Round = 0; Round < Rounds; ++Round) {
    Expected = Expected * 2 + 1;
    Expected += 3;
    Expected *= 2;
    Expected += 1;
  }

  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    if (Data[I] != Expected) {
      std::cout << "Data[" << I << "] == " << Data[I] << ", expected "
                << Expected << std::endl;
      ++Failures;
    }
  }

  std::cout << (Failures == 0 ? "Test passed" : "Test failed") << std::endl;
  return Failures == 0 ? 0 : 1;
}
