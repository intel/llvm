// REQUIRES: linux, gpu && offload
// RUN: %{build} %device_msan_flags -O0 -g -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck %s

// The intercepted zeCommandListAppendMemoryFill only unpoisons what it fills.
#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int data) { return data + 1; }

void check_unfilled_part(sycl::queue &Q) {
  std::cout << "check_unfilled_part" << std::endl;
  auto *Array = sycl::malloc_device<int>(4, Q);
  Q.fill(Array, 7, 2).wait();
  Q.single_task([=]() { Array[0] = check(Array[3]); }).wait();
  sycl::free(Array, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_unfilled_part
// CHECK: use-of-uninitialized-value
// CHECK-NOT: PASS

int main() {
  sycl::queue Q;
  check_unfilled_part(Q);
  return 0;
}
