// REQUIRES: opencl || level_zero || cuda
// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out 2>&1 | FileCheck %s
//
// TODO: Reenable on Windows, see https://github.com/intel/llvm/issues/14768
// XFAIL: windows

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  return 0;
}

// CHECK: urQueueRelease
// CHECK: urContextRelease
