// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=debug:1 %{run} %t 2>&1 | FileCheck --check-prefixes CHECK-DEBUG %s
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=debug:0 %{run} %t 2>&1 | FileCheck %s
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  int *array = sycl::malloc_device<int>(1, Q);

  Q.submit([&](sycl::handler &h) {
    h.single_task<class MyKernel>([=]() { *array = 0; });
  });
  Q.wait();
  // CHECK-DEBUG: [kernel]
  // CHECK-NOT: [kernel]

  std::cout << "PASS" << std::endl;
  return 0;
}
