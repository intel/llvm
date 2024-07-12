// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s

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
       cgh.parallel_for<class Test>(
           sycl::nd_range<1>(size_x / 2 + 1, 1),
           [=](sycl::nd_item<1> item) { accessor[item.get_global_id()] *= 2; });
     }).wait();
    // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Memory Buffer
    // CHECK: {{READ of size 4 at kernel <.*Test> LID\(0, 0, 0\) GID\(8, 0, 0\)}}
    // CHECK: {{#0 .* .*subbuffer.cpp:}}[[@LINE-4]]
  }

  return 0;
}
