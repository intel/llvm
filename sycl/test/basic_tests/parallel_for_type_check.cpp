// RUN: %clangxx -fsycl -fsycl-device-only -D__SYCL_INTERNAL_API -O0 -c -emit-llvm -S -o - %s | FileCheck %s

// This test performs basic type check for sycl::id that is used in result type.

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  sycl::queue q;

  // Initialize data array
  const int sz = 16;
  int data[sz] = {0};
  for (int i = 0; i < sz; ++i) {
    data[i] = i;
  }

  // Check user defined sycl::item wrapper
  sycl::buffer<int> data_buf(data, sz);
  q.submit([&](sycl::handler &h) {
    auto buf_acc = data_buf.get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(
        sycl::range<1>{sz},
        // CHECK: cl{{.*}}sycl{{.*}}detail{{.*}}RoundedRangeKernel{{.*}}id{{.*}}main{{.*}}handler
        [=](sycl::id<1> item) { buf_acc[item] += 1; });
  });
  q.wait();

  return 0;
}
