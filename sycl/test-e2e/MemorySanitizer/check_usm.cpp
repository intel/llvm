// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O0 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

// UNSUPPORTED: cpu
// UNSUPPORTED-TRACKER: CMPLRLLVM-64618

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

__attribute__((noinline)) long long foo(int data1, long long data2) {
  return data1 + data2;
}

void check_memset(sycl::queue &Q) {
  std::cout << "check_memset" << std::endl;
  auto *array = sycl::malloc_device<int>(2, Q);
  auto ev1 = Q.memset(array, 0, 2 * sizeof(int));
  auto ev2 = Q.single_task(ev1, [=]() { array[0] = foo(array[0], array[1]); });
  Q.wait();
  sycl::free(array, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_memset
// CHECK-NOT: use-of-uninitialized-value
// CHECK: PASS

void check_memcpy1(sycl::queue &Q) {
  std::cout << "check_memcpy1" << std::endl;
  auto *source = sycl::malloc_host<int>(2, Q);
  auto *array = sycl::malloc_device<int>(2, Q);
  // FIXME: We don't support shadow propagation on host/shared usm
  auto ev1 = Q.memcpy(array, source, 2 * sizeof(int));
  auto ev2 = Q.single_task(ev1, [=]() { array[0] = foo(array[0], array[1]); });
  Q.wait();
  sycl::free(array, Q);
  sycl::free(source, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_memcpy1
// CHECK-NOT: use-of-uninitialized-value
// CHECK: PASS

void check_memcpy2(sycl::queue &Q) {
  std::cout << "check_memcpy2" << std::endl;
  auto *source = sycl::malloc_device<int>(2, Q);
  auto *array = sycl::malloc_device<int>(2, Q);
  // FIXME: We don't support shadow propagation on host/shared usm
  auto ev1 = Q.memcpy(array, source, 2 * sizeof(int));
  auto ev2 = Q.single_task(ev1, [=]() { array[0] = foo(array[0], array[1]); });
  Q.wait();
  sycl::free(array, Q);
  sycl::free(source, Q);
  std::cout << "PASS" << std::endl;
}
// CHECK-LABEL: check_memcpy2
// CHECK: use-of-uninitialized-value
// CHECK-NOT: PASS

int main() {
  sycl::queue Q;
  check_memset(Q);
  check_memcpy1(Q);
  check_memcpy2(Q);
  return 0;
}
