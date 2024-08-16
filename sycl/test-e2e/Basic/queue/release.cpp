// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s
//
// TODO: Reenable on Windows, see https://github.com/intel/llvm/issues/14768
// XFAIL: hip_nvidia, windows

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
