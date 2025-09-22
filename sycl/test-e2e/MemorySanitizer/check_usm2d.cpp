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
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/usm.hpp>

constexpr size_t Pitch = 4;
constexpr size_t Width = 2;
constexpr size_t Height = 2;
constexpr size_t Size = Pitch * Height;

__attribute__((noinline)) int check(int data1, int data2) {
  return data1 + data2;
}

void check_memcpy2d(sycl::queue &Q) {
  std::cout << "check_memcpy2d" << std::endl;

  auto *source = sycl::malloc_device<int>(Size, Q);
  Q.memset(source, 0, Size * sizeof(int)).wait();

  auto *dest = sycl::malloc_device<int>(Size, Q);
  Q.ext_oneapi_memcpy2d(dest, Pitch * sizeof(int), source, Pitch * sizeof(int),
                        Width * sizeof(int), Height)
      .wait();

  Q.single_task<class Test1>([=]() {
     dest[0] = check(dest[0], dest[1]);
     dest[0] = check(dest[4], dest[5]);
   }).wait();
  // CHECK-NOT: check_usm2d.cpp:[[@LINE-3]]

  Q.single_task<class Test2>([=]() {
     dest[0] = check(dest[2], dest[3]);
   }).wait();
  // CHECK: use-of-uninitialized-value
  // CHECK: check_usm2d.cpp:[[@LINE-3]]

  sycl::free(dest, Q);
  sycl::free(source, Q);
}

int main() {
  sycl::queue Q;
  check_memcpy2d(Q);
  return 0;
}
