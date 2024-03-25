// REQUIRES: opencl

// RUN: %{build} %if cl_options %{/Od%} %else %{-O0%} -o %t0.out
// RUN: %if !acc %{ env SYCL_PI_TRACE=-1 %{run} %t0.out 2>&1 | FileCheck %s --check-prefixes=CHECKOCL0 %}
// RUN: %{build} -O1 -o %t1.out
// RUN: %if !acc %{ env SYCL_PI_TRACE=-1 %{run} %t1.out 2>&1 | FileCheck %s --check-prefixes=CHECKOCL1 %}
// RUN: %{build} -O2 -o %t2.out
// RUN: %if !acc %{ env SYCL_PI_TRACE=-1 %{run} %t2.out 2>&1 | FileCheck %s --check-prefixes=CHECKOCL2 %}
// RUN: %{build} -O3 -o %t3.out
// RUN: %if !acc %{ env SYCL_PI_TRACE=-1 %{run} %t3.out 2>&1 | FileCheck %s --check-prefixes=CHECKOCL3 %}

// RUN: %{build} -O0 -o %t.out
// RUN: %{run} %t.out

// This test verifies the propagation of front-end compiler optimization
// option to the backend.
// API call in device code:
// Following is expected addition of options for opencl backend:
// Front-end option | OpenCL backend option
//       -O0        |    -cl-opt-disable
//       -O1        |    /* no option */
//       -O2        |    /* no option */
//       -O3        |    /* no option */

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) { h.single_task([=]() {}); });
  std::cout << "sycl-optlevel test passed\n";
  return 0;
}

// CHECK-LABEL: ---> piProgramBuild(
// CHECKOCL0: -cl-opt-disable
// CHECKOCL1-NOT: -cl-opt-disable
// CHECKOCL2-NOT: -cl-opt-disable
// CHECKOCL3-NOT: -cl-opt-disable
// CHECK: ) ---> pi_result : PI_SUCCESS
