// RUN: %clang_cc1 -no-opaque-pointers -fsycl-is-device -fintelfpga -triple spir64_fpga -IInputs -emit-llvm %s -o - | FileCheck %s

// This test checks that we generate appropriate code for division
// operations of _BitInts of size greater than 128 bits, since it
// is allowed when -fintelfpga is enabled.

#include "Inputs/sycl.hpp"

// CHECK: define{{.*}} void @_Z3fooDB211_S_(i211 addrspace(4)* {{.*}} sret(i211) align 8 %agg.result, i211* {{.*}} byval(i211) align 8 %[[ARG1:[0-9]+]], i211* {{.*}} byval(i211) align 8 %[[ARG2:[0-9]+]])
signed _BitInt(211) foo(signed _BitInt(211) a, signed _BitInt(211) b) {
  // CHECK: %[[VAR_A:a]] = load i211, i211* %[[ARG1]], align 8
  // CHECK: %[[VAR_B:b]] = load i211, i211* %[[ARG2]], align 8
  // CHECK: %[[RES:div]] = sdiv i211 %[[VAR_A]], %[[VAR_B]]
  // CHECK: store i211 %[[RES]], i211 addrspace(4)* %agg.result, align 8
  // CHECK: ret void
  return a / b;
}

int main() {
  sycl::handler h;
  auto lambda = []() {
    _BitInt(211) a, b = 3, c = 4;
    a = foo(b, c);
  };
  h.single_task(lambda);
}
