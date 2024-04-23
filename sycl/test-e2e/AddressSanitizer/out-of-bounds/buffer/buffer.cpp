// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: %force_device_asan_rt %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t.out
// RUN: %force_device_asan_rt %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t.out
// RUN: %force_device_asan_rt %{run} not %t.out 2>&1 | FileCheck %s

#include "sycl/sycl.hpp"

using namespace sycl;

static const int N = 16;

int main() {
  queue q;

  std::vector<int> v(N);
  for (int i = 0; i < N; i++)
    v[i] = i;

  {
    buffer<int, 1> buf(v.data(), v.size());
    q.submit([&](handler &h) {
       auto A = buf.get_access<access::mode::read_write>(h);
       h.parallel_for<class Test>(range<1>(N + 1), [=](id<1> i) { A[i] *= 2; });
     }).wait();
    // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Memory Buffer
    // CHECK: {{READ of size 4 at kernel <.*Test> LID\(0, 0, 0\) GID\(16, 0, 0\)}}
    // CHECK: {{#0 .* .*buffer.cpp:}}[[@LINE-4]]
  }

  return 0;
}
