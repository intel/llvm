// RUN: polygeist-opt --canonicalize %s | FileCheck %s 
  
// CHECK-LABEL: func.func @foo(%arg0: i1) -> i32 {
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    %0 = arith.select %arg0, %c1_i32, %c2_i32 : i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }

func.func @foo(%arg0: i1) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = arith.select %arg0, %c1_i32, %c2_i32 : i32
  return %0 : i32
}
