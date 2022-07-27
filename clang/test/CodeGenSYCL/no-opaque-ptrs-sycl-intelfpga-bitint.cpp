// RUN: %clang_cc1 -no-opaque-pointers -fsycl-is-device -fintelfpga -triple spir64_fpga -aux-triple x86_64-unknown-linux-gnu -IInputs -emit-llvm %s -o - | FileCheck %s

// This test checks that we generate appropriate code for division
// operations of _BitInts of size greater than 128 bits, since it
// is allowed when -fintelfpga is enabled.  The test uses a value
// of 4096 for the bitsize, the max that is currently supported.

#include "Inputs/sycl.hpp"

// CHECK: define{{.*}} void @_Z3fooDB4096_S_(i4096 addrspace(4)* {{.*}} sret(i4096) align 8 %agg.result, i4096* {{.*}} byval(i4096) align 8 %[[ARG1:[0-9]+]], i4096* {{.*}} byval(i4096) align 8 %[[ARG2:[0-9]+]])
signed _BitInt(4096) foo(signed _BitInt(4096) a, signed _BitInt(4096) b) {
  // CHECK: %[[VAR_A:a]] = load i4096, i4096* %[[ARG1]], align 8
  // CHECK: %[[VAR_B:b]] = load i4096, i4096* %[[ARG2]], align 8
  // CHECK: %[[RES:div]] = sdiv i4096 %[[VAR_A]], %[[VAR_B]]
  // CHECK: store i4096 %[[RES]], i4096 addrspace(4)* %agg.result, align 8
  // CHECK: ret void
  return a / b;
}

int main() {
  sycl::handler h;
  auto lambda = []() {
    _BitInt(4096) a, b = 3, c = 4;
    a = foo(b, c);
  };
  h.single_task(lambda);
}
