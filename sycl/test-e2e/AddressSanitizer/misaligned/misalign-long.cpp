// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O0 -g -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <random>

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(1, 7);

  sycl::queue Q;
  constexpr std::size_t N = 4;
  auto *array = sycl::malloc_shared<long long>(N, Q);
  auto offset = distrib(gen);
  std::cout << "offset: " << offset << std::endl;
  array = (long long *)((char *)array + offset);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(sycl::nd_range<1>(N, 1),
                                   [=](sycl::nd_item<1> item) { ++array[0]; });
    Q.wait();
  });
  // CHECK: ERROR: DeviceSanitizer: misaligned-access on Shared USM
  // CHECK: READ of size 8 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID({{.*}}, 0, 0)
  // CHECK:   #0 {{.*}} {{.*misalign-long.cpp}}:[[@LINE-5]]

  return 0;
}
