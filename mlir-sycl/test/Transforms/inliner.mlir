// RUN: sycl-mlir-opt -sycl-always-inline -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @caller() -> i32 {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32  
// CHECK-NEXT:    %0 = sycl.call() {FunctionName = @callee, MangledFunctionName = @callee, TypeName = @A} : () -> i32
// CHECK-NEXT:    %1 = arith.addi %c1_i32, %0 : i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }

module @host_module {

func.func @caller() -> i32 {
  %res1 = sycl.call() {FunctionName = @"inlinable_callee", MangledFunctionName = @inlinable_callee, TypeName = @A} : () -> i32
  %res2 = sycl.call() {FunctionName = @"callee", MangledFunctionName = @callee, TypeName = @A} : () -> i32  
  %res = arith.addi %res1, %res2 : i32
  return %res : i32
}

func.func @inlinable_callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32 : i32
}

func.func @callee() -> i32 {
  %c2_i32 = arith.constant 2 : i32
  return %c2_i32 : i32
}

}

// -----

// CHECK-LABEL: gpu.func @caller() -> i32 {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32  
// CHECK-NEXT:    %0 = sycl.call() {FunctionName = @callee, MangledFunctionName = @callee, TypeName = @A} : () -> i32
// CHECK-NEXT:    %1 = sycl.call() {FunctionName = @gpu_callee, MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// CHECK-NEXT:    %2 = arith.addi %c1_i32, %0 : i32
// CHECK-NEXT:    %3 = arith.addi %2, %1 : i32
// CHECK-NEXT:    gpu.return %3 : i32
// CHECK-NEXT:  }

gpu.module @module {

gpu.func @caller() -> i32 {
  %res1 = sycl.call() {FunctionName = @"inlinable_callee", MangledFunctionName = @inlinable_callee, TypeName = @A} : () -> i32
  %res2 = sycl.call() {FunctionName = @"callee", MangledFunctionName = @callee, TypeName = @A} : () -> i32  
  %res3 = sycl.call() {FunctionName = @"gpu_callee", MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
  %res4 = arith.addi %res1, %res2 : i32
  %res5 = arith.addi %res4, %res3 : i32  
  gpu.return %res5 : i32
}

func.func @inlinable_callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32 : i32
}

func.func @callee() -> i32 {
  %c2_i32 = arith.constant 2 : i32
  return %c2_i32 : i32
}

gpu.func @gpu_func_callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c3_i32 = arith.constant 3 : i32
  gpu.return %c3_i32 : i32
}

}
