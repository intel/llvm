// RUN: %clang_cc1 -fsycl-is-device -triple spir64-sycldevice -IInputs -emit-llvm %s -o - | FileCheck %s
// This test ensures that we don't generate a kernel before we have instantiated
// all the kernel declarations, which can change the name of a kernel in the
// unnamed kernel case. Previously we would have duplicate manglings of the
// below example, because the marking of 'lambda' as a kernel changed the name
// of the kernel before it.

#include "Inputs/sycl.hpp"

int main() {
  sycl::handler h;
  auto lambda = []() {};
  h.single_task([]() {});
  h.single_task(lambda);
}

// Make sure the kernels are instantiated in the correct order, and named
// correctly.
// CHECK: define{{.*}}spir_kernel void @_ZTSZ4mainEUlvE10001_()
// CHECK: define{{.*}}spir_kernel void @_ZTSZ4mainEUlvE10000_()

