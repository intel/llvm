// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_msan_flags -O0 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t2.out
// RUN: %{run} %t2.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

__attribute__((noinline)) int foo(int data1, int data2) {
  return data1 + data2;
}

void check_memset(sycl::queue &q) {
  std::cout << "check_memset" << std::endl;
  sycl::buffer<int, 1> buf(sycl::range<1>(2));
  const int Pattern = 0;

  q.submit([&](sycl::handler &h) {
     auto array = buf.get_access<sycl::access::mode::read_write>(h);
     h.fill(array, Pattern);
   }).wait();

  q.submit([&](sycl::handler &h) {
     auto array = buf.get_access<sycl::access::mode::read_write>(h);
     h.single_task<class MyKernel1>(
         [=]() { array[0] = foo(array[0], array[1]); });
   }).wait();
  std::cout << "PASS" << std::endl;
  // CHECK-LABEL: check_memset
  // CHECK-NOT: use-of-uninitialized-value
  // CHECK: PASS
}

void check_memcpy(sycl::queue &q) {
  std::cout << "check_memcpy" << std::endl;
  int host[2] = {1, 2};
  sycl::buffer<int, 1> buf1(sycl::range<1>(2));
  sycl::buffer<int, 1> buf2(host, sycl::range<1>(2));

  q.submit([&](sycl::handler &h) {
     auto array1 = buf1.get_access<sycl::access::mode::read_write>(h);
     auto array2 = buf2.get_access<sycl::access::mode::read_write>(h);
     h.copy(array2, array1);
   }).wait();

  q.submit([&](sycl::handler &h) {
     auto array = buf1.get_access<sycl::access::mode::read_write>(h);
     h.single_task<class MyKernel2>(
         [=]() { array[0] = foo(array[0], array[1]); });
   }).wait();
  std::cout << "PASS" << std::endl;
  // CHECK-LABEL: check_memcpy
  // CHECK-NOT: use-of-uninitialized-value
  // CHECK: PASS
}

int main() {
  sycl::queue q;

  check_memset(q);
  check_memcpy(q);

  return 0;
}
