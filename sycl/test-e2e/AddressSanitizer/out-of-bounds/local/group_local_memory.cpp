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
// CHECK: READ of size 4 at kernel {{<.*MyKernel>}} LID(6, 0, 0) GID({{.*}}, 0, 0)
// CHECK:   #0 {{.*}} {{.*group_local_memory.cpp}}:[[@LINE-3]]

int main() {
  sycl::queue Q;
  auto data = sycl::malloc_device<int>(1, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          sycl::multi_ptr<int[N], sycl::access::address_space::local_space>
              ptr = sycl::ext::oneapi::group_local_memory<int[N]>(
                  item.get_group());
          auto &ref = *ptr;
          // NOTE: direct access will be optimized out
          data[0] = check(ref, item.get_local_linear_id() * 2 + 4);
        });
  });
  Q.wait();

  sycl::free(data, Q);
  return 0;
}
