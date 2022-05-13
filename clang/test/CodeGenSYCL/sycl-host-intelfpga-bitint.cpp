// RUN: %clang_cc1 -opaque-pointers -fsycl-is-host -fintelfpga -triple x86_64 -emit-llvm %s -o - | FileCheck %s

// This test checks that we generate appropriate code for division
// operations of _BitInts of size greater than 128 bits, since it
// is allowed when -fintelfpga is enabled.

// CHECK: define{{.*}} void @_Z3fooDB211_S_(ptr {{.*}} sret(i211) align 8 %agg.result, ptr {{.*}} byval(i211) align 8 %[[ARG1:[0-9]+]], ptr {{.*}} byval(i211) align 8 %[[ARG2:[0-9]+]])
signed _BitInt(211) foo(signed _BitInt(211) a, signed _BitInt(211) b) {
  // CHECK: %[[VAR_A:a]].addr = alloca i211, align 8
  // CHECK: %[[VAR_B:b]].addr = alloca i211, align 8
  // CHECK: %[[VAR_A]] = load i211, ptr %[[ARG1]], align 8
  // CHECK: %[[VAR_B]] = load i211, ptr %[[ARG2]], align 8
  // CHECK: store i211 %[[VAR_A]], ptr %[[VAR_A]].addr, align 8
  // CHECK: store i211 %[[VAR_B]], ptr %[[VAR_B]].addr, align 8
  // CHECK: %[[TEMP1:[0-9]+]] = load i211, ptr %[[VAR_A]].addr, align 8
  // CHECK: %[[TEMP2:[0-9]+]] = load i211, ptr %[[VAR_B]].addr, align 8
  // CHECK: %div = sdiv i211 %[[TEMP1]], %[[TEMP2]]
  // CHECK: store i211 %div, ptr %agg.result, align 8
  // CHECK: %[[RES:[0-9+]]] = load i211, ptr %agg.result, align 8
  // CHECK: store i211 %[[RES]], ptr %agg.result, align 8
  // CHECK: ret void
  return a / b;
}
