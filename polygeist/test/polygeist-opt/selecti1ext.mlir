// RUN: polygeist-opt --canonicalize %s | FileCheck %s 
  
// CHECK-LABEL: func.func @test1(%arg0: i1) -> i32 {
// CHECK:         %0 = arith.select %arg0, %c1_i32, %c2_i32 : i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }

func.func @test1(%arg0: i1) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = arith.select %arg0, %c1_i32, %c2_i32 : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @test2(%arg0: i1) -> i32 {
// CHECK:         %0 = arith.xori %arg0, %true : i1
// CHECK-NEXT:    %1 = arith.extui %0 : i1 to i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }

func.func @test2(%arg0: i1) -> i32 {
  %true = arith.constant true
  %false = arith.constant false
  %0 = arith.extui %true : i1 to i32
  %1 = arith.extui %false : i1 to i32
  %2 = arith.select %arg0, %1, %0 : i32
  return %2 : i32
}
