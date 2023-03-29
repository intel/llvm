//==----------- sycl-opt-level.cpp  - DPC++ SYCL on-device test
//---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test verifies the propagation of front-end compiler optimization
// option to the backend.
// API call in device code:
// Following is expected addtion of options:
// Front-end option | OpenCL backend option | L0 backend option
//       -O0        |    -cl-opt-disable    |  -ze-opt-disable
//       -O1        |    /* no option */    |  -ze-opt-level=1
//       -O2        |    /* no option */    |  -ze-opt-level=1
//       -O3        |    /* no option */    |  -ze-opt-level=2

// RUN: %clangxx -O0 -fsycl %s -o %t0.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_TRACE=-1 %t0.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK0
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu SYCL_PI_TRACE=-1 %t0.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECKOCL0
// RUN: %clangxx -O1 -fsycl %s -o %t1.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_TRACE=-1 %t1.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK1
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu SYCL_PI_TRACE=-1 %t1.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECKOCL1
// RUN: %clangxx -O2 -fsycl %s -o %t2.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_TRACE=-1 %t2.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK2
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu SYCL_PI_TRACE=-1 %t2.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECKOCL2
// RUN: %clangxx -O3 -fsycl %s -o %t3.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_PI_TRACE=-1 %t3.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECK3
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu SYCL_PI_TRACE=-1 %t3.out 2>&1 %GPU_CHECK_PLACEHOLDER --check-prefixes=CHECKOCL3

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) { h.single_task([=]() {}); });
  std::cout << "sycl-optlevel test passed\n";
  return 0;
}

// CHECK-LABEL: ---> piProgramBuild(
// CHECK0: -ze-opt-disable
// CHECKOCL0: -cl-opt-disable
// CHECK1: -ze-opt-level=1
// CHECKOCL1-NOT: -cl-opt-disable
// CHECK2: -ze-opt-level=1
// CHECKOCL2-NOT: -cl-opt-disable
// CHECK3: -ze-opt-level=2
// CHECKOCL3-NOT: -cl-opt-disable
// CHECK: ) ---> pi_result : PI_SUCCESS
