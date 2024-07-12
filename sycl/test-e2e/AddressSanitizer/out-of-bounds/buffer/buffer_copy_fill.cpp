// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <numeric>

static const int N = 16;

int main() {
  sycl::queue q;

  std::vector<int> v(N);
  std::iota(v.begin(), v.end(), 0);

  {
    sycl::buffer<int, 1> buf(v.size());

    q.submit([&](sycl::handler &h) {
       auto A = buf.get_access<sycl::access::mode::write>(h);
       h.copy(&v[0], A);
     }).wait();

    q.submit([&](sycl::handler &h) {
       auto A = buf.get_access<sycl::access::mode::write>(h);
       h.fill(A, 1);
     }).wait();

    q.submit([&](sycl::handler &h) {
       auto A = buf.get_access<sycl::access::mode::read_write>(h);
       h.parallel_for<class Test>(
           sycl::nd_range<1>(N + 1, 1),
           [=](sycl::nd_item<1> item) { A[item.get_global_id()] *= 2; });
     }).wait();
    // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Memory Buffer
    // CHECK: {{READ of size 4 at kernel <.*Test> LID\(0, 0, 0\) GID\(16, 0, 0\)}}
    // CHECK: {{#0 .* .*buffer_copy_fill.cpp:}}[[@LINE-4]]
  }

  return 0;
}
