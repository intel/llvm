// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=1 %{run} %t.out | FileCheck %s
//
// XFAIL: hip_nvidia

#include <sycl/detail/core.hpp>
int main() {
  sycl::queue q;

  q.single_task<class test>([]() {});
  // no wait. Ensure resources are released anyway.

  return 0;
}

// CHECK: ---> urEnqueueKernelLaunch(
// FIXME the order of these 2 varies between plugins due to a Level Zero
// specific queue workaround.
// CHECK-DAG: ---> urEventRelease(
// CHECK-DAG: ---> urQueueRelease(
// CHECK: ---> urContextRelease(
// CHECK: ---> urKernelRelease(
// CHECK: ---> urProgramRelease(
// CHECK: ---> urDeviceRelease(
