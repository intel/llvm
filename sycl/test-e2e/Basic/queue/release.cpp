// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
//
// XFAIL: hip_nvidia

#include <sycl/sycl.hpp>
int main() {
  sycl::queue q;

  q.single_task<class test>([]() {});
  // no wait. Ensure resources are released anyway.

  return 0;
}

// CHECK: ---> piEnqueueKernelLaunch(
// FIXME the order of these 2 varies between plugins due to a Level Zero
// specific queue workaround.
// CHECK-DAG: ---> piEventRelease(
// CHECK-DAG: ---> piQueueRelease(
// CHECK: ---> piContextRelease(
// CHECK: ---> piKernelRelease(
// CHECK: ---> piProgramRelease(
// CHECK: ---> piDeviceRelease(
