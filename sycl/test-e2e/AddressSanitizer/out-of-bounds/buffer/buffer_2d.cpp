// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -O0 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_sanitizer_flags -O1 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s

#include "sycl/sycl.hpp"

int main() {
  constexpr size_t size_x = 5;
  constexpr size_t size_y = 6;

  std::vector<int> v(size_x * size_y);

  sycl::buffer<int, 2> buf(v.data(), sycl::range<2>(size_x, size_y));

  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
     auto accessor = buf.get_access<sycl::access::mode::read_write>(cgh);
     cgh.parallel_for<class Test>(
         sycl::range<2>(size_x, size_y + 1),
         [=](sycl::id<2> idx) { accessor[idx] = idx[0] + idx[1]; });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Memory Buffer
  // CHECK: {{WRITE of size 4 at kernel <.*Test> LID\(0, 0, 0\) GID\(6, 4, 0\)}}
  // CHECK: {{#0 .* .*buffer_2d.cpp:}}[[@LINE-4]]

  return 0;
}
