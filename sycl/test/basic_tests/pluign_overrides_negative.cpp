// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_OVERRIDE_PI_OPENCL=opencl_test env SYCL_OVERRIDE_PI_LEVEL_ZERO=l0_test env SYCL_OVERRIDE_PI_CUDA=cuda_test env SYCL_OVERRIDE_PI_ROCM=rocm_test env SYCL_PI_TRACE=-1 %t.out > %t.log 2>&1
// RUN: FileCheck %s --input-file %t.log

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  return 0;
}

// CHECK: SYCL_PI_TRACE[all]: Check if plugin is present. Failed to load plugin: opencl_test
// CHECK: SYCL_PI_TRACE[all]: Check if plugin is present. Failed to load plugin: l0_test
// CHECK: SYCL_PI_TRACE[all]: Check if plugin is present. Failed to load plugin: cuda_test
// CHECK: SYCL_PI_TRACE[all]: Check if plugin is present. Failed to load plugin: rocm_test
