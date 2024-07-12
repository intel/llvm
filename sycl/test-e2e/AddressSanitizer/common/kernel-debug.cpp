// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=debug:1 %{run} %t 2>&1 | FileCheck --check-prefixes CHECK-DEBUG %s
// RUN: env SYCL_PREFER_UR=1 UR_LAYER_ASAN_OPTIONS=debug:0 %{run} %t 2>&1 | FileCheck %s
#include <sycl/usm.hpp>

/// This test is used to check enabling/disabling kernel debug message
/// We always use "[kernel]" prefix in kernel debug message

constexpr std::size_t N = 4;
constexpr std::size_t group_size = 1;

int main() {
  sycl::queue Q;
  int *array = sycl::malloc_device<int>(N, Q);

  Q.submit([&](sycl::handler &cgh) {
    auto acc = sycl::local_accessor<int>(group_size, cgh);
    cgh.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          array[item.get_global_id()] = acc[item.get_local_id()];
        });
  });
  Q.wait();
  // CHECK-DEBUG: [kernel]
  // CHECK-NOT: [kernel]

  sycl::free(array, Q);
  std::cout << "PASS" << std::endl;
  return 0;
}
