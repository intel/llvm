// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------- host_task_dependency.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host tasks around a cross-context dependency. A host task runs on the host
// and has no context of its own, so it is never the command a proxy event is
// created for; a proxy is only ever created for a command that goes to a device
// of a different context. Both directions have to keep working:
//
//   1. a host task depending on a kernel of another context, and
//   2. a kernel depending on a host task submitted to a queue of another
//      context.

#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

constexpr size_t N = 128;
constexpr int Repeats = 4096;

int main() {
  sycl::device Dev;

  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev};
  sycl::queue Q2{Ctx2, Dev};

  std::vector<int> Data(N, 0);
  int SeenByHostTask = -1;
  {
    sycl::buffer<int, 1> Buf(Data.data(), sycl::range<1>(N));

    // Produced in Ctx1.
    sycl::event Produce = Q1.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::write_only, sycl::no_init};
      CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
        int Value = 0;
        for (int It = 0; It < Repeats; ++It)
          Value += static_cast<int>(I) + 1;
        Acc[I] = Value / Repeats;
      });
    });

    // A host task on a queue of Ctx2, depending on the kernel of Ctx1.
    sycl::event Host = Q2.submit([&](sycl::handler &CGH) {
      CGH.depends_on(Produce);
      auto Acc = Buf.get_host_access(CGH);
      CGH.host_task([=, &SeenByHostTask]() {
        SeenByHostTask = Acc[N - 1];
        for (size_t I = 0; I < N; ++I)
          Acc[I] += 1;
      });
    });

    // A kernel in Ctx1 depending on the host task submitted to the Ctx2 queue.
    Q1.submit([&](sycl::handler &CGH) {
        CGH.depends_on(Host);
        sycl::accessor Acc{Buf, CGH, sycl::read_write};
        CGH.parallel_for(sycl::range<1>(N),
                         [=](sycl::id<1> I) { Acc[I] *= 2; });
      }).wait_and_throw();

    Q1.wait_and_throw();
    Q2.wait_and_throw();
  }

  int Failures = 0;
  if (SeenByHostTask != static_cast<int>(N)) {
    std::cout << "the host task saw " << SeenByHostTask << ", expected " << N
              << std::endl;
    ++Failures;
  }
  for (size_t I = 0; I < N; ++I) {
    const int Expected = (static_cast<int>(I) + 2) * 2;
    if (Data[I] != Expected) {
      std::cout << "Data[" << I << "] == " << Data[I] << ", expected "
                << Expected << std::endl;
      ++Failures;
    }
  }

  std::cout << (Failures == 0 ? "Test passed" : "Test failed") << std::endl;
  return Failures == 0 ? 0 : 1;
}
