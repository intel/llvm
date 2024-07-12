// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s

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
       h.parallel_for<class Test>(
           sycl::nd_range<1>(N + 1, 1),
           [=](sycl::nd_item<1> item) { A[item.get_global_id()] *= 2; });
     }).wait();
    // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Memory Buffer
    // CHECK: {{READ of size 4 at kernel <.*Test> LID\(0, 0, 0\) GID\(16, 0, 0\)}}
    // CHECK: {{#0 .* .*buffer.cpp:}}[[@LINE-4]]
  }

  return 0;
}
