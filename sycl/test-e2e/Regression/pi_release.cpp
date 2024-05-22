// REQUIRES: opencl || level_zero || cuda
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  return 0;
}

// CHECK: piQueueRelease
// CHECK: piContextRelease
