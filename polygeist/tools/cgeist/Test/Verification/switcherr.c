// RUN: cgeist %s --function=foo -S | FileCheck %s

int foo(int t) {
  int n = 10;
  switch (t) {
  case 1:
    n = 20;
    break;
  case 2:
    n = 30;
    break;
  default:
    return -1;
  }
  return n;
}

// TODO the select should be canonicalized better
// CHECK:   func @foo(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c-1_i32 = arith.constant -1 : i32
// CHECK-DAG:     %c30_i32 = arith.constant 30 : i32
// CHECK-DAG:     %false = arith.constant false
// CHECK-DAG:     %c20_i32 = arith.constant 20 : i32
// CHECK-DAG:     %c10_i32 = arith.constant 10 : i32
// CHECK-DAG:     %true = arith.constant true
// CHECK-DAG:     %0 = llvm.mlir.undef : i32
// CHECK-DAG:     switch %arg0 : i32, [
// CHECK-NEXT:       default: ^bb1(%c10_i32, %false, %c-1_i32 : i32, i1, i32),
// CHECK-NEXT:       1: ^bb1(%c20_i32, %true, %0 : i32, i1, i32),
// CHECK-NEXT:       2: ^bb1(%c30_i32, %true, %0 : i32, i1, i32)
// CHECK-NEXT:     ]
// CHECK-NEXT:   ^bb1(%1: i32, %2: i1, %3: i32):  // 3 preds: ^bb0, ^bb0, ^bb0
// CHECK-NEXT:     %4 = arith.select %2, %1, %3 : i32
// CHECK-NEXT:     return %4 : i32
// CHECK-NEXT:   }
