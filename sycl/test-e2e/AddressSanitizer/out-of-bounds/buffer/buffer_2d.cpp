// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

int main() {
  constexpr size_t size_x = 5;
  constexpr size_t size_y = 6;

  std::vector<int> v(size_x * size_y);

  // We intentionally test sycl::buffer uses host ptr here because in unified
  // runtime we intercept sycl::buffer with usm, we need to cover that pattern
  // here.
  sycl::buffer<int, 2> buf(v.data(), sycl::range<2>(size_x, size_y));

  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
     auto accessor = buf.get_access<sycl::access::mode::read_write>(cgh);
     cgh.parallel_for<class Test>(
         sycl::nd_range<2>({size_x, size_y + 1}, {1, 1}),
         [=](sycl::nd_item<2> item) {
           accessor[item.get_global_id()] = item.get_global_linear_id();
         });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Memory Buffer
  // CHECK: {{WRITE of size 4 at kernel <.*Test> LID\(0, 0, 0\) GID\(6, 4, 0\)}}
  // CHECK: {{#0 .* .*buffer_2d.cpp:}}[[@LINE-5]]

  return 0;
}
