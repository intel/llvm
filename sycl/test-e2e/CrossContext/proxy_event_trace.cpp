// REQUIRES: level_zero_v2_adapter

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s

//==------------------- proxy_event_trace.cpp ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------==//
//
// Checks the mechanism itself rather than its effect: the single cross-context
// dependency below makes the runtime create a host-signalled event in the
// consuming context and signal it once the producer of the other context has
// retired. The two UR calls have to show up in that order in the trace.
//
// The host-signalled event extension is only implemented by the Level Zero v2
// adapter, hence the REQUIRES above; on any other backend the runtime falls
// back to connecting the two commands with a host task and neither call is
// made.

#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

constexpr size_t N = 128;

int main() {
  sycl::device Dev;

  sycl::context Ctx1{Dev};
  sycl::context Ctx2{Dev};
  sycl::queue Q1{Ctx1, Dev};
  sycl::queue Q2{Ctx2, Dev};

  std::vector<int> Data(N, 0);
  {
    sycl::buffer<int, 1> Buf(Data.data(), sycl::range<1>(N));

    sycl::event Produce = Q1.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::write_only, sycl::no_init};
      CGH.parallel_for(sycl::range<1>(N), [=](sycl::id<1> I) {
        Acc[I] = static_cast<int>(I) + 1;
      });
    });

    // The one and only cross-context dependency of this program.
    Q2.submit([&](sycl::handler &CGH) {
        CGH.depends_on(Produce);
        sycl::accessor Acc{Buf, CGH, sycl::read_write};
        CGH.parallel_for(sycl::range<1>(N),
                         [=](sycl::id<1> I) { Acc[I] *= 2; });
      }).wait_and_throw();
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

// A proxy event is created in the consuming context...
// CHECK: ---> urEventCreateHostSignalExp
// ...and signalled once the dependency of the other context has retired.
// CHECK: ---> urEventHostSignalExp
// CHECK: Test passed
