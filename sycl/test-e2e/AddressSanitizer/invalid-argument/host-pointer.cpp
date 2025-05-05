// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env UR_LAYER_ASAN_OPTIONS="detect_kernel_arguments:1" %{run} %t 2>&1 | FileCheck --check-prefixes %if cpu %{ CHECK-CPU %} %if gpu %{ CHECK-GPU %} %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue Q;
  auto *array = sycl::malloc_device<uintptr_t>(1, Q);
  auto hostPtr = new int;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { array[0] = (uintptr_t)hostPtr; });
  });
  Q.wait();

  // CHECK-GPU: ERROR: DeviceSanitizer: invalid-argument
  // CHECK-GPU: The {{[0-9]+}}th argument {{.*}} is not a USM pointer
  // CHECK-CPU-NOT: ERROR: DeviceSanitizer: invalid-argument

  sycl::free(array, Q);
  delete hostPtr;
  puts("PASS\n");
  return 0;
}
