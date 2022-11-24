// RUN: sycl-mlir-opt -always-inline -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @callee() 
// CHECK-SAME     attributes {passthrough = \["alwaysinline"\]} -> i32 {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32  
// CHECK-NEXT:    return %c1_i32
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @caller() -> i32 {
// CHECK-NEXT:    %res = sycl.call() {FunctionName = @"callee", MangledFunctionName = @callee, TypeName = @A} : () -> i32
// CHECK-NEXT:    return %res
// CHECK-NEXT:  }

func.func @callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c1_i32 = arith.constant 1 : i32  
  return %c1_i32 : i32
}

func.func @caller() -> i32 {
  %res = sycl.call() {FunctionName = @"callee", MangledFunctionName = @callee, TypeName = @A} : () -> i32
  return %res : i32
}
