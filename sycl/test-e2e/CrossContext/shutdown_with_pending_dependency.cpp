// REQUIRES: aspect-usm_device_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==-------------- shutdown_with_pending_dependency.cpp --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A cross-context dependency that is still pending when the program ends. The
// consuming command has been handed to the device and is waiting for a
// connection to the producer of another context; that connection is a proxy
// event that somebody has to signal, or the submission never retires and
// shutdown blocks on it forever.
//
// Everything here is leaked on purpose: a queue, context or USM allocation
// going out of scope would wait for the work first, and then there would be
// nothing left pending to shut down with. The test passes by terminating - a
// failure shows up as a hang (a lit timeout) or as a crash during shutdown.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <iostream>

constexpr size_t N = 1024;
// Enough work that the producer is still running when main returns.
constexpr int Repeats = 1 << 16;

int main() {
  sycl::device Dev;

  // Deliberately never destroyed, see above.
  auto *Ctx1 = new sycl::context{Dev};
  auto *Ctx2 = new sycl::context{Dev};
  auto *Q1 = new sycl::queue{*Ctx1, Dev};
  auto *Q2 = new sycl::queue{*Ctx2, Dev};

  int *Dev1 = sycl::malloc_device<int>(N, Dev, *Ctx1);
  int *Dev2 = sycl::malloc_device<int>(N, Dev, *Ctx2);

  sycl::event Produce = Q1->parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
    int Value = 0;
    for (int It = 0; It < Repeats; ++It)
      Value += static_cast<int>(I) + 1;
    Dev1[I] = Value / Repeats;
  });

  // Submitted to a queue of another context and left in flight.
  Q2->submit([&](sycl::handler &CGH) {
    CGH.depends_on(Produce);
    CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) { Dev2[I] = 1; });
  });

  std::cout << "Test passed" << std::endl;
  return 0;
}
