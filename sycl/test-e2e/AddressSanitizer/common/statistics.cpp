// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: %{run} %t 2>&1 | FileCheck %s
// RUN: env UR_LAYER_ASAN_OPTIONS=print_stats:1 %{run} %t 2>&1 | FileCheck --check-prefixes CHECK-STATS %s
#include <sycl/usm.hpp>

/// This test is used to check enabling/disabling memory overhead statistics
/// We always use "Stats" prefix in statistics message like asan

constexpr std::size_t N = 4;
constexpr std::size_t group_size = 1;

int main() {
  sycl::queue Q;
  int *array = sycl::malloc_device<int>(1024 * 1024, Q);

  Q.submit([&](sycl::handler &cgh) {
    auto acc = sycl::local_accessor<int>(group_size, cgh);
    cgh.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          array[item.get_global_id()] = acc[item.get_local_id()];
        });
  });
  Q.wait();
  // CHECK-STATS: Stats
  // CHECK-NOT: Stats

  sycl::free(array, Q);
  std::cout << "PASS" << std::endl;
  return 0;
}
