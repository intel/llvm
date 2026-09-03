// REQUIRES: aspect-usm_device_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------------- usm_event_dependency.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A cross-context dependency with no buffer involved at all: a USM allocation
// belongs to a single context, so the two contexts here hand data over through
// plain host memory and the only thing ordering them is the explicit dependency
// on a foreign event.
//
// That makes the ordering directly observable: if the copy out of host memory
// in the second context does not wait for the copy into it in the first, it
// reads the initial values instead of the produced ones.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <iostream>
#include <vector>

constexpr size_t N = 4096;
constexpr int Repeats = 4096;
constexpr int Sentinel = -1;

int main() {
  sycl::device Dev;

  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev};
  sycl::queue Q2{Ctx2, Dev};

  int *Dev1 = sycl::malloc_device<int>(N, Dev, Ctx1);
  int *Dev2 = sycl::malloc_device<int>(N, Dev, Ctx2);

  // Handed over between the two contexts. Filled with a value that no kernel
  // below produces, so a copy that runs too early is visible in the result.
  std::vector<int> Handover(N, Sentinel);
  std::vector<int> Result(N, 0);

  // Ctx1: produce into its own allocation, then copy the result out to host
  // memory.
  sycl::event Produce = Q1.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
    int Value = 0;
    for (int It = 0; It < Repeats; ++It)
      Value += static_cast<int>(I) + 1;
    Dev1[I] = Value / Repeats;
  });
  sycl::event CopyOut =
      Q1.memcpy(Handover.data(), Dev1, N * sizeof(int), {Produce});

  // Ctx2: the copy in must not start before the copy out has finished, and the
  // event it has to wait for belongs to Ctx1.
  sycl::event CopyIn =
      Q2.memcpy(Dev2, Handover.data(), N * sizeof(int), {CopyOut});
  sycl::event Consume = Q2.submit([&](sycl::handler &CGH) {
    CGH.depends_on(CopyIn);
    CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) { Dev2[I] *= 2; });
  });
  Q2.memcpy(Result.data(), Dev2, N * sizeof(int), {Consume}).wait_and_throw();

  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    const int Expected = (static_cast<int>(I) + 1) * 2;
    if (Result[I] != Expected) {
      std::cout << "Result[" << I << "] == " << Result[I] << ", expected "
                << Expected << std::endl;
      ++Failures;
    }
  }

  sycl::free(Dev1, Ctx1);
  sycl::free(Dev2, Ctx2);

  std::cout << (Failures == 0 ? "Test passed" : "Test failed") << std::endl;
  return Failures == 0 ? 0 : 1;
}
