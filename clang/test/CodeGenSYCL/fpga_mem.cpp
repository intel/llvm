// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -fsycl-unique-prefix=THE_PREFIX -std=c++17 -opaque-pointers -emit-llvm %s -o - | FileCheck %s
#include "sycl.hpp"

// Test cases below show that ...

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
queue q;

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_name_1>([=]() {
      [[intel::doublepump, intel::fpga_memory("MLAB")]]int b [10];
      // fpga_mem<int[10], decltype(properties(clock_2x_true))> a;
      fpga_mem<int[10]> a;
    });
  });
}


// CHECK: ...
