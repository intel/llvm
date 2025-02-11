// REQUIRES: linux, cpu
// RUN: %{build} %device_asan_flags -O0 -g -fPIC -shared -DSHARED_LIB -o %t.so
// RUN: %{build} %device_asan_flags -O0 -g -fPIC -Wl,-rpath,. %t.so -o %t
// RUN: %{run} %t 2>&1 | FileCheck %s
#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

#if defined(SHARED_LIB)
void test(sycl::queue &Q, size_t N) {
  auto *array = sycl::malloc_device<char>(N, Q);

  Q.submit([&](sycl::handler &h) {
     h.parallel_for<class MyKernelR_4>(
         sycl::nd_range<1>(N, 1),
         [=](sycl::nd_item<1> item) { ++array[item.get_global_id(0)]; });
   }).wait();
}

#else

void test(sycl::queue &Q, size_t N);

int main() {
  sycl::queue Q;
  constexpr std::size_t N = 12;

  test(Q, N);

  // CHECK: ERROR: DeviceSanitizer: detected memory leaks of Device USM
  // CHECK: Direct leak of 12 byte(s) at {{0x.*}} allocated from:
  // CHECK: in test{{.*memory-leak-shared-lib.cpp:}}[[@LINE-21]]

  return 0;
}

#endif
