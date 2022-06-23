// RUN: %clang_cc1 -no-opaque-pointers -fsycl-is-host -fintelfpga -triple x86_64 -aux-triple spir64_fpga -emit-llvm %s -o - | FileCheck %s

// This test checks that we generate appropriate code for division
// operations of _BitInts of size greater than 128 bits, since it
// is allowed when -fintelfpga is enabled.  The test uses a value of
// 2048 for the bitsize as that is the maximum that is currently
// supported.

// CHECK: define{{.*}} void @_Z3fooDB2048_S_(i2048* {{.*}} sret(i2048) align 8 %agg.result, i2048* {{.*}} byval(i2048) align 8 %[[ARG1:[0-9]+]], i2048* {{.*}} byval(i2048) align 8 %[[ARG2:[0-9]+]])
signed _BitInt(2048) foo(signed _BitInt(2048) a, signed _BitInt(2048) b) {
  // CHECK: %[[VAR_A:a]].addr = alloca i2048, align 8
  // CHECK: %[[VAR_B:b]].addr = alloca i2048, align 8
  // CHECK: %[[VAR_A]] = load i2048, i2048* %[[ARG1]], align 8
  // CHECK: %[[VAR_B]] = load i2048, i2048* %[[ARG2]], align 8
  // CHECK: store i2048 %[[VAR_A]], i2048* %[[VAR_A]].addr, align 8
  // CHECK: store i2048 %[[VAR_B]], i2048* %[[VAR_B]].addr, align 8
  // CHECK: %[[TEMP1:[0-9]+]] = load i2048, i2048* %[[VAR_A]].addr, align 8
  // CHECK: %[[TEMP2:[0-9]+]] = load i2048, i2048* %[[VAR_B]].addr, align 8
  // CHECK: %div = sdiv i2048 %[[TEMP1]], %[[TEMP2]]
  // CHECK: store i2048 %div, i2048* %agg.result, align 8
  // CHECK: %[[RES:[0-9+]]] = load i2048, i2048* %agg.result, align 8
  // CHECK: store i2048 %[[RES]], i2048* %agg.result, align 8
  // CHECK: ret void
  return a / b;
}
