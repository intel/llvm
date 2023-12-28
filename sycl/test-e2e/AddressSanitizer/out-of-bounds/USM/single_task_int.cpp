// REQUIRES: linux
// RUN: %{build} %device_sanitizer_flags -g -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %s

#include <sycl/sycl.hpp>

const int N = 1024;

int main() {
  sycl::queue Q;

  int *data = sycl::malloc_device<int>(N, Q);

  // CHECK: DeviceSanitizer: out-of-bounds-access on USM Device Memory
  Q.single_task([=]() {
    for (int i = 0; i <= N; ++i) {
      data[i] = i; // <== buffer-overflow here
    }
  });

  Q.wait();
  return 0;
}
