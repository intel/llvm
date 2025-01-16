// REQUIRES: linux, cpu || (gpu && level_zero)
// REQUIRES: spir64
// RUN: %{build} %device_msan_flags -O0 -g -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_msan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/device_global/device_global.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

sycl::ext::oneapi::experimental::device_global<
    int[4], decltype(properties(device_image_scope, host_access_read_write))>
    dev_global;

__attribute__((noinline)) int check(int data) { return data + 1; }

int main() {
  sycl::queue Q;
  int *array = sycl::malloc_device<int>(4, Q);

  Q.submit([&](sycl::handler &h) {
     h.single_task<class Test1>([=]() {
       dev_global[0] = 42;
       array[0] = check(dev_global[1]);
       array[1] = dev_global[1];
     });
   }).wait();

  int val[4];
  Q.copy(dev_global, val).wait();
  assert(val[0] == 42);

  Q.submit([&](sycl::handler &h) {
     h.single_task<class Test2>([=]() {
       array[0] = check(array[1]);
       dev_global[1] = array[2]; // uninitialzed value
     });
   }).wait();

  Q.submit([&](sycl::handler &h) {
     h.single_task<class Test3>([=]() {
       array[0] = dev_global[1];
       check(array[0]);
     });
   }).wait();
  // CHECK: use-of-uninitialized-value
  // CHECK-NEXT: kernel <{{.*Test3}}>

  sycl::free(array, Q);

  return 0;
}
