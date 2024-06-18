// REQUIRES: level_zero

// RUN: %{build} %if cl_options %{/Od%} %else %{-O0%} -o %t0.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t0.out 2>&1 | FileCheck %s --check-prefixes=CHECK0
// RUN: %{build} -O1 -o %t1.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t1.out 2>&1 | FileCheck %s --check-prefixes=CHECK1
// RUN: %{build} -O2 -o %t2.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t2.out 2>&1 | FileCheck %s --check-prefixes=CHECK1
// RUN: %{build} -O3 -o %t3.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t3.out 2>&1 | FileCheck %s --check-prefixes=CHECK1

// This test verifies the propagation of front-end compiler optimization
// option to the backend.
// API call in device code:
// Following is expected addition of options for level_zero backend:
// Front-end option | L0 backend option
//       -O0        |    -ze-opt-disable
//       -O1        |    -ze-opt-level=2
//       -O2        |    -ze-opt-level=2
//       -O3        |    -ze-opt-level=2

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) { h.single_task([=]() {}); });
  std::cout << "sycl-optlevel test passed\n";
  return 0;
}

// CHECK-LABEL: ---> piProgramBuild(
// CHECK0: -ze-opt-disable
// CHECK1: -ze-opt-level=2
// CHECK: ) ---> pi_result : PI_SUCCESS
