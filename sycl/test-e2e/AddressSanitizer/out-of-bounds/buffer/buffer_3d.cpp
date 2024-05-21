// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: %force_device_asan_rt %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t.out
// RUN: %force_device_asan_rt %{run} not %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

int main() {
  constexpr size_t size_x = 5;
  constexpr size_t size_y = 6;
  constexpr size_t size_z = 7;

  std::vector<int> v(size_x * size_y * size_z);

  sycl::buffer<int, 3> buf(v.data(), sycl::range<3>(size_x, size_y, size_z));

  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
     auto accessor = buf.get_access<sycl::access::mode::read_write>(cgh);

     cgh.parallel_for<class Test>(
         sycl::nd_range<3>({size_x, size_y, size_z + 1}, {1, 1, 1}),
         [=](sycl::nd_item<3> item) {
           accessor[item.get_global_id()] =
               item.get_global_id(0) * item.get_global_range(1) *
                   item.get_global_range(2) +
               item.get_global_id(1) * item.get_global_range(2) +
               item.get_global_id(2);
         });
   }).wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Memory Buffer
  // CHECK: {{WRITE of size 4 at kernel <.*Test> LID\(0, 0, 0\) GID\(7, 5, 4\)}}
  // CHECK: {{#0 .* .*buffer_3d.cpp:}}[[@LINE-9]]

  return 0;
}
