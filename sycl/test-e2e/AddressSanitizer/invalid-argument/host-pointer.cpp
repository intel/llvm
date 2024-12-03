// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS="detect_kernel_arguments:1" %{run} %if gpu %{ not %} %t 2>&1 | FileCheck --check-prefixes %if cpu %{ CHECK-CPU %} %if gpu %{ CHECK-GPU %} %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

///
/// GPU devices don't support shared system USM currently, so passing host
/// pointer to kernel is invalid.
/// CPU devices support shared system USM.
///

int main() {
  sycl::queue Q;
  auto hostPtr = new int;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { *hostPtr = 0; });
  });
  Q.wait();

  // CHECK-GPU: ERROR: DeviceSanitizer: invalid-argument
  // CHECK-GPU: The {{[0-9]+}}th argument {{.*}} is not a USM pointer
  // CHECK-CPU-NOT: ERROR: DeviceSanitizer: invalid-argument

  delete hostPtr;
  puts("PASS\n");
  return 0;
}
