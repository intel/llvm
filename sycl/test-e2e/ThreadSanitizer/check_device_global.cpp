// REQUIRES: linux, cpu || (gpu && level_zero)
// ALLOW_RETRIES: 10
// RUN: %{build} %device_tsan_flags -O0 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_tsan_flags -O2 -g -o %t2.out
// RUN: %{run} %t2.out 2>&1 | FileCheck %s
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/device_global/device_global.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

sycl::ext::oneapi::experimental::device_global<
    int[4], decltype(properties(device_image_scope, host_access_read_write))>
    dev_global;

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class Test>(sycl::nd_range<1>(128, 8),
                                [=](sycl::nd_item<1>) { dev_global[0]++; });
   }).wait();
  // CHECK: WARNING: DeviceSanitizer: data race
  // CHECK-NEXT: When write of size 4 at 0x{{.*}} in kernel <{{.*}}Test>
  // CHECK-NEXT: #0 {{.*}}check_device_global.cpp:[[@LINE-4]]

  return 0;
}
