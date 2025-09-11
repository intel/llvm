// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O0 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

// UNSUPPORTED: cpu
// UNSUPPORTED-TRACKER: CMPLRLLVM-64618

// XFAIL: spirv-backend && gpu
// XFAIL-TRACKER: CMPLRLLVM-64705

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) int check(int data1, int data2) {
  return data1 + data2;
}

void check_memset(sycl::queue &Q) {
  std::cout << "check_memset" << std::endl;
  auto *array = sycl::malloc_device<int>(2, Q);
  auto ev1 = Q.memset(array, 0, 2 * sizeof(int));
  auto ev2 =
      Q.single_task(ev1, [=]() { array[0] = check(array[0], array[1]); });
  Q.wait();
  sycl::free(array, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_memset
// CHECK-NOT: use-of-uninitialized-value
// CHECK: PASS

void check_fill(sycl::queue &Q) {
  std::cout << "check_fill" << std::endl;
  int *array = sycl::malloc_device<int>(2, Q);
  uint32_t pattern = 0;
  auto ev1 = Q.fill(array, pattern, 2);
  auto ev2 =
      Q.single_task(ev1, [=]() { array[0] = check(array[0], array[1]); });
  Q.wait();
  sycl::free(array, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_fill
// CHECK-NOT: use-of-uninitialized-value
// CHECK: PASS

void check_memcpy(sycl::queue &Q) {
  std::cout << "check_memcpy" << std::endl;
  auto *source = sycl::malloc_device<int>(2, Q);
  auto *array = sycl::malloc_device<int>(2, Q);
  auto ev1 = Q.memcpy(array, source, 2 * sizeof(int));
  auto ev2 =
      Q.single_task(ev1, [=]() { array[0] = check(array[0], array[1]); });
  Q.wait();
  sycl::free(array, Q);
  sycl::free(source, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_memcpy
// CHECK: use-of-uninitialized-value
// CHECK-NOT: PASS

int main() {
  sycl::queue Q;
  check_memset(Q);
  check_fill(Q);
  check_memcpy(Q);
  return 0;
}
