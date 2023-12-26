// REQUIRES: linux
// UNSUPPORTED: true
// TODO: rely on "printf_abort"

// RUN: %{build} %device_sanitizer_flags -g -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s

// XFAIL: *

#include <sycl/sycl.hpp>

const int N = 1024;

int main() {
  sycl::queue Q;

  int *data = sycl::malloc_device<int>(N, Q);

  // CHECK: DeviceSanitizer: out-of-bounds-access on USM Device Memory
  Q.single_task([=]() {
    int i = N;
    // infinite loop
    while (true) {
      data[--i] = i; // <== buffer-underflow here
    }
  });

  Q.wait();
  return 0;
}