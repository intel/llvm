// RUN: cgeist %s --function=foo -S | FileCheck %s

#define N 8
int foo() {
  static int bar[N];
  return bar[0];
}

// CHECK-DAG:   memref.global "private" @"foo@static@bar" : memref<8xi32> = dense<0> {alignment = 16 : i64}
// CHECK-LABEL:   func @foo() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = memref.get_global @"foo@static@bar" : memref<8xi32>
// CHECK-NEXT:     %1 = affine.load %0[0] : memref<8xi32>
// CHECK-NEXT:     return %1 : i32
// CHECK-NEXT:   }
