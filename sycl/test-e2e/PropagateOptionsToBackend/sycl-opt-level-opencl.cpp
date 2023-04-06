// REQUIRES: opencl

// RUN: %clangxx -O0 -fsycl %s -o %t0.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu SYCL_PI_TRACE=-1 %t0.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECKOCL0
// RUN: %clangxx -O1 -fsycl %s -o %t1.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu SYCL_PI_TRACE=-1 %t1.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECKOCL1
// RUN: %clangxx -O2 -fsycl %s -o %t2.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu SYCL_PI_TRACE=-1 %t2.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECKOCL2
// RUN: %clangxx -O3 -fsycl %s -o %t3.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu SYCL_PI_TRACE=-1 %t3.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECKOCL3

// This test verifies the propagation of front-end compiler optimization
// option to the backend.
// API call in device code:
// Following is expected addition of options for opencl backend:
// Front-end option | OpenCL backend option
//       -O0        |    -cl-opt-disable
//       -O1        |    /* no option */
//       -O2        |    /* no option */
//       -O3        |    /* no option */

#include <sycl/sycl.hpp>

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
