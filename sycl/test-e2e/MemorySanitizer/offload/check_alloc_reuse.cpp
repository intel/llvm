// REQUIRES: linux, gpu && offload
// RUN: %{build} %device_msan_flags -O0 -g -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck %s

// Every allocation must reach the intercepted zeMemAlloc*: memory from
// liboffload's pool has no shadow mapped (GPU page fault), so the layer
// disables the pool.
#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int data) { return data + 1; }

int main() {
  sycl::queue Q;
  auto *Out = sycl::malloc_device<int>(1, Q);
  Q.memset(Out, 0, sizeof(int)).wait();

  for (int I = 0; I < 2; ++I) {
    auto *Tmp = sycl::malloc_device<int>(4, Q);
    Q.memset(Tmp, 0, 4 * sizeof(int)).wait();
    Q.single_task([=]() { Out[0] = check(Tmp[3]); }).wait();
    sycl::free(Tmp, Q);
  }
  std::cout << "PASS" << std::endl;
  // CHECK-NOT: use-of-uninitialized-value
  // CHECK: PASS

  auto *Array = sycl::malloc_device<int>(4, Q);
  Q.single_task([=]() { Out[0] = check(Array[3]); }).wait();
  // CHECK: use-of-uninitialized-value

  sycl::free(Array, Q);
  sycl::free(Out, Q);
  return 0;
}
