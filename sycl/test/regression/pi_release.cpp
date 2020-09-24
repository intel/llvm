// REQUIRES: cpu
// RUN: %clangxx -fsycl  -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %CPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck %s

#include <CL/sycl.hpp>

int main() {
  sycl::queue q;
  return 0;
}

// CHECK: piQueueRelease
// CHECK: piContextRelease
