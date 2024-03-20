// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t.out 2>&1 | FileCheck %s
#include <sycl/sycl.hpp>

constexpr std::size_t N = 16;
constexpr std::size_t group_size = 8;

int main() {
  sycl::queue Q;
  auto *data = sycl::malloc_host<int>(1, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          sycl::multi_ptr<int[N], sycl::access::address_space::local_space>
              ptr = sycl::ext::oneapi::group_local_memory<int[N]>(
                  item.get_group());
          auto &ref = *ptr;
          ref[item.get_local_linear_id() * 2 + 4] = 42;
          // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Local Memory
          // CHECK: {{WRITE of size 4 at kernel <.*MyKernel> LID\(6, 0, 0\) GID\(.*, 0, 0\)}}
          // CHECK: {{  #0 .* .*local-overflow-1.cpp:}}[[@LINE-3]]
        });
  });

  Q.wait();
  return 0;
}
