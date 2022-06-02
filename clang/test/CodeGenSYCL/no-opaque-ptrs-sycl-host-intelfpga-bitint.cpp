// RUN: %clang_cc1 -no-opaque-pointers -fsycl-is-host -fintelfpga -triple x86_64 -emit-llvm %s -o - | FileCheck %s

// This test checks that we generate appropriate code for division
// operations of _BitInts of size greater than 128 bits, since it
// is allowed when -fintelfpga is enabled.

// CHECK: define{{.*}} void @_Z3fooDB211_S_(i211* {{.*}} sret(i211) align 8 %agg.result, i211* {{.*}} byval(i211) align 8 %[[ARG1:[0-9]+]], i211* {{.*}} byval(i211) align 8 %[[ARG2:[0-9]+]])
signed _BitInt(211) foo(signed _BitInt(211) a, signed _BitInt(211) b) {
  // CHECK: %[[VAR_A:a]].addr = alloca i211, align 8
  // CHECK: %[[VAR_B:b]].addr = alloca i211, align 8
  // CHECK: %[[VAR_A]] = load i211, i211* %[[ARG1]], align 8
  // CHECK: %[[VAR_B]] = load i211, i211* %[[ARG2]], align 8
  // CHECK: store i211 %[[VAR_A]], i211* %[[VAR_A]].addr, align 8
  // CHECK: store i211 %[[VAR_B]], i211* %[[VAR_B]].addr, align 8
  // CHECK: %[[TEMP1:[0-9]+]] = load i211, i211* %[[VAR_A]].addr, align 8
  // CHECK: %[[TEMP2:[0-9]+]] = load i211, i211* %[[VAR_B]].addr, align 8
  // CHECK: %div = sdiv i211 %[[TEMP1]], %[[TEMP2]]
  // CHECK: store i211 %div, i211* %agg.result, align 8
  // CHECK: %[[RES:[0-9+]]] = load i211, i211* %agg.result, align 8
  // CHECK: store i211 %[[RES]], i211* %agg.result, align 8
  // CHECK: ret void
  return a / b;
}
