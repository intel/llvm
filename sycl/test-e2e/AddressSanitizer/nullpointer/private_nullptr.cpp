// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_asan_flags -O0 -g -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

// FIXME: There's an issue in gfx driver, so this test pending here.
// XFAIL: *

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/address_cast.hpp>

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 4;
  int *array = nullptr;

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, 1), [=](sycl::nd_item<1> item) {
          auto private_array =
              sycl::ext::oneapi::experimental::static_address_cast<
                  sycl::access::address_space::private_space>(array);
          private_array[0] = 0;
        });
    Q.wait();
  });
  // CHECK: ERROR: DeviceSanitizer: null-pointer-access on Unknown Memory
  // CHECK: WRITE of size 4 at kernel {{<.*MyKernel>}} LID(0, 0, 0) GID({{.*}}, 0, 0)
  // CHECK: {{.*private_nullptr.cpp}}:[[@LINE-5]]

  return 0;
}
