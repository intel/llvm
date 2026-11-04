// REQUIRES: linux, gpu && offload
// RUN: %{build} %device_msan_flags -O0 -g -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck %s

// Shadow propagation of the intercepted zeCommandListAppendMemoryCopy.
#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int data) { return data + 1; }

void check_host_to_device(sycl::queue &Q) {
  std::cout << "check_host_to_device" << std::endl;
  int Host[4] = {1, 2, 3, 4};
  auto *Array = sycl::malloc_device<int>(4, Q);
  Q.memcpy(Array, Host, sizeof(Host)).wait();
  Q.single_task([=]() { Array[0] = check(Array[3]); }).wait();
  sycl::free(Array, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_host_to_device
// CHECK-NOT: use-of-uninitialized-value
// CHECK: PASS

void check_copied_part(sycl::queue &Q) {
  std::cout << "check_copied_part" << std::endl;
  auto *Source = sycl::malloc_device<int>(2, Q);
  auto *Array = sycl::malloc_device<int>(4, Q);
  Q.memset(Source, 0, 2 * sizeof(int)).wait();
  Q.memcpy(Array, Source, 2 * sizeof(int)).wait();
  Q.single_task([=]() { Array[0] = check(Array[1]); }).wait();
  sycl::free(Array, Q);
  sycl::free(Source, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_copied_part
// CHECK-NOT: use-of-uninitialized-value
// CHECK: PASS

void check_uncopied_part(sycl::queue &Q) {
  std::cout << "check_uncopied_part" << std::endl;
  auto *Source = sycl::malloc_device<int>(2, Q);
  auto *Array = sycl::malloc_device<int>(4, Q);
  Q.memset(Source, 0, 2 * sizeof(int)).wait();
  Q.memcpy(Array, Source, 2 * sizeof(int)).wait();
  Q.single_task([=]() { Array[0] = check(Array[2]); }).wait();
  sycl::free(Array, Q);
  sycl::free(Source, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_uncopied_part
// CHECK: use-of-uninitialized-value
// CHECK-NOT: PASS

int main() {
  sycl::queue Q;
  check_host_to_device(Q);
  check_copied_part(Q);
  check_uncopied_part(Q);
  return 0;
}
