// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

__attribute__((noinline)) long long foo(int data1, long long data2) {
  return data1 + data2;
}

int main() {
  sycl::queue q;
  int data1[1];
  long long data2[1];

  {
    sycl::buffer<int, 1> buf1(data1, sycl::range<1>(1));
    sycl::buffer<long long, 1> buf2(data2, sycl::range<1>(1));
    q.submit([&](sycl::handler &h) {
       auto array1 = buf1.get_access<sycl::access::mode::read_write>(h);
       auto array2 = buf2.get_access<sycl::access::mode::read_write>(h);
       h.single_task<class MyKernel>(
           [=]() { array1[0] = foo(array1[0], array2[0]); });
     }).wait();
    // CHECK: use-of-uninitialized-value
    // CHECK: kernel <{{.*MyKernel}}>
    // CHECK: #0 {{.*}} {{.*check_buffer_host_ptr.cpp}}:[[@LINE-4]]
  }

  return 0;
}
