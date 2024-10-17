// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s %if !windows %{--check-prefixes=CHECK-RELEASE%}
//
// XFAIL: hip_nvidia

#include <sycl/detail/core.hpp>
int main() {
  sycl::queue q;

  q.single_task<class test>([]() {});
  // no wait. Ensure resources are released anyway.

  return 0;
}

// CHECK: <--- urEnqueueKernelLaunch(
// FIXME the order of these 2 varies between adapters due to a Level Zero
// specific queue workaround.
// CHECK-DAG: <--- urEventRelease(
// CHECK-DAG: <--- urQueueRelease(

// On Windows, dlls unloading is inconsistent and if we try to release these UR
// objects manually, inconsistent hangs happen due to a race between unloading
// the UR adapters dlls (in addition to their dependency dlls) and the releasing
// of these UR objects. So, we currently shutdown without releasing them and
// windows should handle the memory cleanup.

// CHECK-RELEASE: <--- urContextRelease(
// CHECK-RELEASE: <--- urKernelRelease(
// CHECK-RELEASE: <--- urProgramRelease(
// CHECK-RELEASE: <--- urDeviceRelease(
