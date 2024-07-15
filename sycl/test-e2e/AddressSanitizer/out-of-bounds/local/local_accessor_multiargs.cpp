// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -g -O0 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -g -O1 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -g -O2 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
#include <sycl/usm.hpp>

constexpr std::size_t N = 8;
constexpr std::size_t group_size = 4;

int main() {
  sycl::queue Q;
  auto data1 = sycl::malloc_device<int>(N, Q);
  auto data2 = sycl::malloc_device<int>(N, Q);

  Q.submit([&](sycl::handler &cgh) {
    auto acc1 = sycl::local_accessor<int>(group_size, cgh);
    auto acc2 = sycl::local_accessor<int>(group_size, cgh);
    auto acc3 = sycl::local_accessor<int>(group_size, cgh);
    cgh.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          data1[item.get_global_id()] = acc1[item.get_local_id()] +
                                        acc2[item.get_local_id()] +
                                        acc3[item.get_local_id()];
          data2[item.get_global_id()] = acc1[item.get_local_id()] +
                                        acc2[item.get_local_id() + 1] +
                                        acc3[item.get_local_id()];
        });
    // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Local Memory
    // CHECK: READ of size 4 at kernel {{<.*MyKernel>}} LID(3, 0, 0) GID({{.*}}, 0, 0)
    // CHECK:   #0 {{.*}} {{.*local_accessor_multiargs.cpp}}:[[@LINE-5]]
  });

  Q.wait();
  sycl::free(data1, Q);
  sycl::free(data2, Q);
  return 0;
}
