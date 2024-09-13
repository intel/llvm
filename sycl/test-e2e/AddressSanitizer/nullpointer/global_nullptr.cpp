// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O0 -g -o %t
// RUN: %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t
// RUN: %{run} not %t 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t
// RUN: %{run} not %t 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 4;
  int *array = nullptr;

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, 1), [=](sycl::nd_item<1> item) {
          auto private_array =
              sycl::ext::oneapi::experimental::static_address_cast<
                  sycl::access::address_space::private_space,
                  sycl::access::decorated::no>(array);
          private_array[0] = 0;
        });
    Q.wait();
  });
  // CHECK: ERROR: DeviceSanitizer: null-pointer-access on Unknown Memory
  // CHECK: WRITE of size 4 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID({{.*}}, 0, 0)
  // CHECK: {{.*global_nullptr.cpp}}:[[@LINE-5]]

  return 0;
}
