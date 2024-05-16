// REQUIRES: linux
// RUN: %{build} %device_asan_flags -g -O0 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -g -O1 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -g -O2 -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/usm.hpp>

constexpr std::size_t N = 16;
constexpr std::size_t group_size = 8;

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          sycl::multi_ptr<int[N], sycl::access::address_space::local_space>
              ptr = sycl::ext::oneapi::group_local_memory<int[N]>(
                  item.get_group());
          auto &ref = *ptr;
          ref[item.get_local_linear_id() * 2 + 4] = 42;
          // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Local Memory
          // CHECK: WRITE of size 4 at kernel {{<.*MyKernel>}} LID(6, 0, 0) GID({{.*}}, 0, 0)
          // CHECK:   #0 {{.*}} {{.*group_local_memory.cpp}}:[[@LINE-3]]
        });
  });

  Q.wait();
  return 0;
}
