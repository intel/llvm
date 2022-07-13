// REQUIRES: opencl || level_zero || cuda
// RUN: %clangxx -fsycl  -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out 2>&1 %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=-1 %ACC_RUN_PLACEHOLDER %t.out 2>&1 %ACC_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=-1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  return 0;
}

// CHECK: piQueueRelease
// CHECK: piContextRelease
