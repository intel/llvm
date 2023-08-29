// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test cases below show that ...

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental; // for fpga_mem
namespace oneapi = sycl::ext::oneapi::experimental; // for properties

int main() {
  queue Q;
  int f = 5;

  Q.single_task([=]() {
    // [[intel::num_banks(888)]]int a [10];
    intel::fpga_mem<int[10], decltype(oneapi::properties(oneapi::num_banks<888>))> a;
    volatile int ReadVal = a[f];
  });
  return 0;
}


// CHECK: ...
