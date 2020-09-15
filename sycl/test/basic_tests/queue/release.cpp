// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out | FileCheck %s


#include <CL/sycl.hpp>
int main() {
  sycl::queue q;

  q.single_task<class test>([]() {});
  // no wait.

  return 0;
}

//CHECK: ---> piEnqueueKernelLaunch(
//CHECK: ---> piEventRelease(
//CHECK: ---> piQueueRelease(
//CHECK: ---> piContextRelease(
//CHECK: ---> piKernelRelease(
//CHECK: ---> piProgramRelease(
