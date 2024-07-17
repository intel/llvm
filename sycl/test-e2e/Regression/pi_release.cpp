// REQUIRES: opencl || level_zero || cuda
// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=1 %{run} %t.out 2>&1 | FileCheck %s
// See github issue https://github.com/intel/llvm/issues/14598
// XFAIL: windows

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  return 0;
}

// CHECK: urQueueRelease
// CHECK: urContextRelease
