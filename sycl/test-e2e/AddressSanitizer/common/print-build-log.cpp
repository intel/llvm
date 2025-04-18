// This test is adapted from sycl/test-e2e/KernelAndProgram/build-log.cpp
// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -DGPU -o %t_gpu.out
// RUN: %{build} %device_asan_flags -o %t.out
// RUN: %{run} not --crash %if gpu %{ %t_gpu.out %} %else %{ %t.out %} 2>&1 | FileCheck %s

#include <iostream>
#include <sycl/detail/core.hpp>
SYCL_EXTERNAL
void symbol_that_does_not_exist();

void test() {
  sycl::queue Queue;

  // Submitting this kernel should result in an exception with error code
  // `sycl::errc::build` and a message indicating
  // "PI_ERROR_BUILD_PROGRAM_FAILURE".
  auto Kernel = []() {
#ifdef __SYCL_DEVICE_ONLY__
#ifdef GPU
    asm volatile("undefined\n");
#else  // GPU
    symbol_that_does_not_exist();
#endif // GPU
#endif // __SYCL_DEVICE_ONLY__
  };

  Queue.submit(
      [&](sycl::handler &CGH) { CGH.single_task<class SingleTask>(Kernel); });
}

// CHECK: <SANITIZER>[ERROR]: Printing build log for program

int main() {
  test();
  return 0;
}
