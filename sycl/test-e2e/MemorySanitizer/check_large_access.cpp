// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

#include "sycl/detail/core.hpp"
#include <sycl/vector.hpp>

int main() {
  sycl::buffer<sycl::int3, 1> b(sycl::range<1>(2));
  sycl::queue myQueue;
  myQueue.submit([&](sycl::handler &cgh) {
    auto B = b.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<class MyKernel>(sycl::range<1>{2}, [=](sycl::id<1> ID) {
      B[ID] = sycl::int3{(sycl::int3)ID[0]} / B[ID];
    });
  }).wait();
  // CHECK: use-of-uninitialized-value
  // CHECK: kernel <{{.*MyKernel}}>

  return 0;
}