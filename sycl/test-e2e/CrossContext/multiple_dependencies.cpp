// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------- multiple_dependencies.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One command depending on three commands, each from a different foreign
// context. Every one of those dependencies has to be resolved separately - each
// gets a proxy event of its own in the consuming context - and the consumer
// must wait for all of them.

#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

constexpr size_t N = 256;
constexpr int Repeats = 4096;

int main() {
  sycl::device Dev;

  // A context per producer, plus one for the consumer.
  sycl::context Ctx1{Dev}, Ctx2{Dev}, Ctx3{Dev}, Ctx4{Dev};
  sycl::queue Q1{Ctx1, Dev}, Q2{Ctx2, Dev}, Q3{Ctx3, Dev}, Q4{Ctx4, Dev};

  std::vector<int> A(N, 0), B(N, 0), C(N, 0), Sum(N, 0);
  {
    sycl::buffer<int, 1> ABuf(A.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> BBuf(B.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> CBuf(C.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> SumBuf(Sum.data(), sycl::range<1>(N));

    auto produce = [](sycl::queue &Q, sycl::buffer<int, 1> &Buf, int Factor) {
      return Q.submit([&, Factor](sycl::handler &CGH) {
        sycl::accessor Acc{Buf, CGH, sycl::write_only, sycl::no_init};
        CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
          int Value = 0;
          for (int It = 0; It < Repeats; ++It)
            Value += (static_cast<int>(I) + 1) * Factor;
          Acc[I] = Value / Repeats;
        });
      });
    };

    std::vector<sycl::event> Events{produce(Q1, ABuf, 1), produce(Q2, BBuf, 2),
                                    produce(Q3, CBuf, 3)};

    // Every dependency of this command crosses a context boundary, both as an
    // explicit event dependency and as a buffer requirement.
    Q4.submit([&](sycl::handler &CGH) {
        CGH.depends_on(Events);
        sycl::accessor InA{ABuf, CGH, sycl::read_only};
        sycl::accessor InB{BBuf, CGH, sycl::read_only};
        sycl::accessor InC{CBuf, CGH, sycl::read_only};
        sycl::accessor Out{SumBuf, CGH, sycl::write_only, sycl::no_init};
        CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
          Out[I] = InA[I] + InB[I] + InC[I];
        });
      }).wait_and_throw();
  }

  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    // Factors 1 + 2 + 3.
    const int Expected = (static_cast<int>(I) + 1) * 6;
    if (Sum[I] != Expected) {
      std::cout << "Sum[" << I << "] == " << Sum[I] << ", expected " << Expected
                << std::endl;
      ++Failures;
    }
  }

  std::cout << (Failures == 0 ? "Test passed" : "Test failed") << std::endl;
  return Failures == 0 ? 0 : 1;
}
