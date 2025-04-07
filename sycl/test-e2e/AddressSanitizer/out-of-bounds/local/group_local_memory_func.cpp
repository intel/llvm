// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -g -O0 -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -g -O1 -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -g -O2 -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/usm.hpp>

constexpr std::size_t N = 16;
constexpr std::size_t group_size = 8;

__attribute__((noinline)) int check(int *ref, int index) { return ref[index]; }
// CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Local Memory
// CHECK: READ of size 4 at kernel {{<.*MyKernel>}} LID({{.*}}, 0, 0) GID({{.*}}, 0, 0)
// CHECK:   #0 {{.*}} {{.*group_local_memory_func.cpp}}:[[@LINE-3]]

__attribute__((noinline)) int test_local(sycl::nd_item<1> &item) {
  auto local_mem =
      sycl::ext::oneapi::group_local_memory<int[group_size]>(item.get_group());
  // NOTE: direct access will be optimized out
  return check(*local_mem, group_size);
}

int main() {
  sycl::queue Q;
  auto data = sycl::malloc_device<int>(N, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size),
        [=](sycl::nd_item<1> item) { data[0] = test_local(item); });
  });
  Q.wait();

  sycl::free(data, Q);
  return 0;
}
