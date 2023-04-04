// REQUIRES: gpu, level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: env ZE_DEBUG=1 SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL=0 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER

// Check that the fill operation is using compute (0 ordinal) engine.
//
// CHECK: [getZeQueue]: create queue ordinal = 0
// CHECKL ZE ---> zeCommandListAppendMemoryFill

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

const int N = 1024;

int main() {
  auto Q = queue(gpu_selector_v);
  auto p = malloc_device(N, Q);
  Q.memset(p, 1, N).wait();
  sycl::free(p, Q);

  return 0;
}
