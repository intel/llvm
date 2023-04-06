// REQUIRES: level_zero

// RUN: %clangxx -O0 -fsycl %s -o %t0.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_TRACE=-1 %t0.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK0
// RUN: %clangxx -O1 -fsycl %s -o %t1.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_TRACE=-1 %t1.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK1
// RUN: %clangxx -O2 -fsycl %s -o %t2.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_TRACE=-1 %t2.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK2
// RUN: %clangxx -O3 -fsycl %s -o %t3.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_TRACE=-1 %t3.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK3

// This test verifies the propagation of front-end compiler optimization
// option to the backend.
// API call in device code:
// Following is expected addition of options for level_zero backend:
// Front-end option | L0 backend option
//       -O0        |    -ze-opt-disable
//       -O1        |    -ze-opt-level=1
//       -O2        |    -ze-opt-level=1
//       -O3        |    -ze-opt-level=2

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) { h.single_task([=]() {}); });
  std::cout << "sycl-optlevel test passed\n";
  return 0;
}

// CHECK-LABEL: ---> piProgramBuild(
// CHECK0: -ze-opt-disable
// CHECK1: -ze-opt-level=1
// CHECK2: -ze-opt-level=1
// CHECK3: -ze-opt-level=2
// CHECK: ) ---> pi_result : PI_SUCCESS
