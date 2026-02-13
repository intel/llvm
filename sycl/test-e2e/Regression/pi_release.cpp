// REQUIRES: opencl || level_zero || cuda
// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out %if !windows %{2>&1 | FileCheck %s %}

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  return 0;
}

// On Windows, dlls unloading is inconsistent and if we try to release these UR
// objects manually, inconsistent hangs happen due to a race between unloading
// the UR adapters dlls (in addition to their dependency dlls) and the releasing
// of these UR objects. So, we currently shutdown without releasing them and
// windows should handle the memory cleanup.

// CHECK: <--- urQueueRelease
// CHECK: <--- urContextRelease
