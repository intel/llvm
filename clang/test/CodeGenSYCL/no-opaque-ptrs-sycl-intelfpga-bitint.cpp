// RUN: %clang_cc1 -no-opaque-pointers -fsycl-is-device -fintelfpga -triple spir64_fpga -aux-triple x86_64-unknown-linux-gnu -IInputs -emit-llvm %s -o - | FileCheck %s

// This test checks that we generate appropriate code for division
// operations of _BitInts of size greater than 128 bits, since it
// is allowed when -fintelfpga is enabled.  The test uses a value
// of 2048 for the bitsize, the max that is currently supported.

#include "Inputs/sycl.hpp"

// CHECK: define{{.*}} void @_Z3fooDB2048_S_(i2048 addrspace(4)* {{.*}} sret(i2048) align 8 %agg.result, i2048* {{.*}} byval(i2048) align 8 %[[ARG1:[0-9]+]], i2048* {{.*}} byval(i2048) align 8 %[[ARG2:[0-9]+]])
signed _BitInt(2048) foo(signed _BitInt(2048) a, signed _BitInt(2048) b) {
  // CHECK: %[[VAR_A:a]] = load i2048, i2048* %[[ARG1]], align 8
  // CHECK: %[[VAR_B:b]] = load i2048, i2048* %[[ARG2]], align 8
  // CHECK: %[[RES:div]] = sdiv i2048 %[[VAR_A]], %[[VAR_B]]
  // CHECK: store i2048 %[[RES]], i2048 addrspace(4)* %agg.result, align 8
  // CHECK: ret void
  return a / b;
}

int main() {
  sycl::handler h;
  auto lambda = []() {
    _BitInt(2048) a, b = 3, c = 4;
    a = foo(b, c);
  };
  h.single_task(lambda);
}
