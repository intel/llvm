// REQUIRES: linux
// RUN: %{build} %device_asan_flags -g -O0 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -g -O1 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -g -O2 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
#include <sycl/usm.hpp>

constexpr std::size_t N = 4;
constexpr std::size_t group_size = 1;

int main() {
  sycl::queue Q;
  auto data = sycl::malloc_device<int>(N, Q);

  Q.submit([&](sycl::handler &cgh) {
    auto acc = sycl::local_accessor<int>(group_size, cgh);
    cgh.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          data[item.get_global_id()] = acc[item.get_local_id() + 1];
          // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Local Memory
          // CHECK: READ of size 4 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID({{.*}}, 0, 0)
          // CHECK:   #0 {{.*}} {{.*local_accessor_basic.cpp}}:[[@LINE-3]]
        });
  });

  Q.wait();
  sycl::free(data, Q);
  return 0;
}
