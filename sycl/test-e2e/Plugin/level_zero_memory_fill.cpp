// REQUIRES: gpu, level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s
// RUN: env UR_L0_DEBUG=1 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL=0 %{run} %t.out 2>&1 | FileCheck %s

// Check that the fill operation is using compute (0 ordinal) engine.
//
// CHECK: [getZeQueue]: create queue ordinal = 0
// CHECKL ZE ---> zeCommandListAppendMemoryFill

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

const int N = 1024;

int main() {
  auto Q = queue(gpu_selector_v);
  auto p = malloc_device(N, Q);
  Q.memset(p, 1, N).wait();
  sycl::free(p, Q);

  return 0;
}
