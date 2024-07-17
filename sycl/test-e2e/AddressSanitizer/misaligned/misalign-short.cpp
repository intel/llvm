// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: env SYCL_PREFER_UR=1 %{run} not %t 2>&1 | FileCheck %s
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <random>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 4;
  auto *array = sycl::malloc_device<short>(N, Q);
  array = (short *)((char *)array + 1);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(sycl::nd_range<1>(N, 1),
                                   [=](sycl::nd_item<1> item) { ++array[0]; });
    Q.wait();
  });
  // CHECK: ERROR: DeviceSanitizer: misaligned-access on Device USM
  // CHECK: READ of size 2 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID({{.*}}, 0, 0)
  // CHECK:   #0 {{.*}} {{.*misalign-short.cpp}}:[[@LINE-5]]

  return 0;
}
