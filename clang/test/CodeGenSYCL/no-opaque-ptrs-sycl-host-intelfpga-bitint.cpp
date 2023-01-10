// RUN: %clang_cc1 -no-opaque-pointers -fsycl-is-host -fintelfpga -triple x86_64 -aux-triple spir64_fpga -emit-llvm %s -o - | FileCheck %s

// This test checks that we generate appropriate code for division
// operations of _BitInts of size greater than 128 bits, since it
// is allowed when -fintelfpga is enabled.  The test uses a value of
// 4096 for the bitsize as that is the maximum that is currently
// supported.

// CHECK: define{{.*}} void @_Z3fooDB4096_S_(i4096* {{.*}} sret(i4096) align 8 %agg.result, i4096* {{.*}} byval(i4096) align 8 %[[ARG1:[0-9]+]], i4096* {{.*}} byval(i4096) align 8 %[[ARG2:[0-9]+]])
signed _BitInt(4096) foo(signed _BitInt(4096) a, signed _BitInt(4096) b) {
  // CHECK: %[[VAR_A:a]].addr = alloca i4096, align 8
  // CHECK: %[[VAR_B:b]].addr = alloca i4096, align 8
  // CHECK: %[[VAR_A]] = load i4096, i4096* %[[ARG1]], align 8
  // CHECK: %[[VAR_B]] = load i4096, i4096* %[[ARG2]], align 8
  // CHECK: store i4096 %[[VAR_A]], i4096* %[[VAR_A]].addr, align 8
  // CHECK: store i4096 %[[VAR_B]], i4096* %[[VAR_B]].addr, align 8
  // CHECK: %[[TEMP1:[0-9]+]] = load i4096, i4096* %[[VAR_A]].addr, align 8
  // CHECK: %[[TEMP2:[0-9]+]] = load i4096, i4096* %[[VAR_B]].addr, align 8
  // CHECK: %div = sdiv i4096 %[[TEMP1]], %[[TEMP2]]
  // CHECK: store i4096 %div, i4096* %agg.result, align 8
  // CHECK: %[[RES:[0-9+]]] = load i4096, i4096* %agg.result, align 8
  // CHECK: store i4096 %[[RES]], i4096* %agg.result, align 8
  // CHECK: ret void
  return a / b;
}
