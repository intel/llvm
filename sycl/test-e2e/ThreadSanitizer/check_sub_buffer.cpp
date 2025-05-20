// REQUIRES: linux, cpu || (gpu && level_zero)
// ALLOW_RETRIES: 10
// RUN: %{build} %device_tsan_flags -O0 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_tsan_flags -O2 -g -o %t2.out
// RUN: %{run} %t2.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

int main() {
  constexpr size_t size_x = 16;

  std::vector<int> v(size_x);
  for (size_t i = 0; i < size_x; i++)
    v[i] = i;

  {
    sycl::queue q;
    sycl::buffer<int> buf(v.data(), v.size());
    sycl::buffer<int> sub_buf(buf, {size_x / 2}, {size_x / 2});

    q.submit([&](sycl::handler &cgh) {
       auto accessor = sub_buf.get_access<sycl::access::mode::read_write>(cgh);
       cgh.parallel_for<class Test>(sycl::nd_range<1>(size_x / 2, 1),
                                    [=](sycl::nd_item<1>) { accessor[0]++; });
     }).wait();
    // CHECK: WARNING: DeviceSanitizer: data race
    // CHECK-NEXT: When write of size 4 at 0x{{.*}} in kernel <{{.*}}Test>
    // CHECK-NEXT: #0 {{.*}}check_sub_buffer.cpp:[[@LINE-4]]
  }

  return 0;
}
