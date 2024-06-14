// RUN: %{build}  -o %t.out
// RUN: env SYCL_PI_TRACE=-1 %{run} %t.out | FileCheck %s

// flaky on OCL GPU, CPU and FPGA.
// Seems to sometimes just terminate everything when piProgramRelease is being called.
// the PI Trace seems right. We create exactly one program and it has
// one call to piProgramRetain followed by two calls to piProgramRelease.
// Yet OpenCL is sometimes crashing on that second call on Windows

// UNSUPPORTED: opencl && windows

//
// XFAIL: hip_nvidia

#include <sycl/detail/core.hpp>
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
// CHECK: ---> piTearDown(
