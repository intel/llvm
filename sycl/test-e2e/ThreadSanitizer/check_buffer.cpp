// REQUIRES: linux, cpu || (gpu && level_zero)
// ALLOW_RETRIES: 10
// RUN: %{build} %device_tsan_flags -O0 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_tsan_flags -O2 -g -o %t2.out
// RUN: %{run} %t2.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

static const int N = 16;

int main() {
  sycl::queue q;

  std::vector<int> v(N);

  {
    // We intentionally test sycl::buffer uses host ptr and trigger data write
    // back here because in unified runtime we intercept sycl::buffer with usm,
    // we need to cover that pattern here.
    sycl::buffer<int, 1> buf(v.data(), v.size());
    q.submit([&](sycl::handler &h) {
       auto A = buf.get_access<sycl::access::mode::read_write>(h);
       h.parallel_for<class Test>(sycl::nd_range<1>(N, 1),
                                  [=](sycl::nd_item<1>) { A[0]++; });
     }).wait();
    // CHECK: WARNING: DeviceSanitizer: data race
    // CHECK-NEXT: When write of size 4 at 0x{{.*}} in kernel <{{.*}}Test>
    // CHECK-NEXT: #0 {{.*}}check_buffer.cpp:[[@LINE-4]]
  }

  return 0;
}
