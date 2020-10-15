// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_THROW_ON_BLOCK=1 SYCL_DEVICE_TYPE=HOST %t.out | FileCheck %s
// RUN: env SYCL_THROW_ON_BLOCK=1 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_THROW_ON_BLOCK=1 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_THROW_ON_BLOCK=1 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER

// Simple test to check that asynchronous stream flushing works when flag is
// enabled to throw an exception on attempt to enqueue a blocked command.

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  queue Queue;

  Queue.submit([&](handler &CGH) {
    stream Out(1024, 80, CGH);
    CGH.parallel_for<class auto_flush1>(
        range<1>(2), [=](id<1> i) { Out << "Hello World!" << endl; });
  });
  Queue.wait();
  // CHECK: Hello World!
  // CHECK-NEXT: Hello World!

  return 0;
}
