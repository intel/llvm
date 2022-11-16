// RUN:  %clang_cc1 -O1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -emit-llvm %s -o - | FileCheck %s

// Test that address spaces are deduced correctly by compiler optimizations.

#include "sycl.hpp"

using namespace sycl;

void foo(const float *usm_in, float* usm_out) {
  queue Q;
  Q.submit([&](handler &cgh) {
    cgh.single_task<class test>([=](){
      *usm_out = *usm_in;
    });
  });
}

// No addrspacecast before loading and storing values
// CHECK-NOT: addrspacecast
// CHECK:  %0 = load float, ptr addrspace(1)
// CHECK:  store float %0, ptr addrspace(1)
