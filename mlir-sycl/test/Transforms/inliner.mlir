// RUN: sycl-mlir-opt -split-input-file -inliner="mode=alwaysinline remove-dead-callees=true" -verify-diagnostics -mlir-pass-statistics %s 2>&1 | FileCheck --check-prefix=ALWAYS-INLINE %s
// RUN: sycl-mlir-opt -split-input-file -inliner="mode=simple remove-dead-callees=true" -verify-diagnostics -mlir-pass-statistics %s 2>&1 | FileCheck --check-prefix=INLINE %s

// COM: Ensure a func.func can be inlined in a func.func caller iff the callee is 'alwaysinline'.
// COM: Ensure a gpu.func cannot be inlined in a func.func caller (even if it has the 'alwaysinline' attribute).

// ALWAYS-INLINE-LABEL: InlinePass
// ALWAYS-INLINE-LABEL:  (S) 1 num-inlined-calls - Number of inlined calls

// ALWAYS-INLINE-LABEL: func.func @caller() -> i32 {
// ALWAYS-INLINE-NEXT:    %c1_i32 = arith.constant 1 : i32
// ALWAYS-INLINE-NEXT:    %0 = sycl.call() {FunctionName = @inline_hint_callee_, MangledFunctionName = @inline_hint_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %1 = sycl.call() {FunctionName = @private_callee_, MangledFunctionName = @private_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %2 = sycl.call() {FunctionName = @callee_, MangledFunctionName = @callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %3 = sycl.call() {FunctionName = @gpu_func_callee_, MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %4 = arith.addi %c1_i32, %0 : i32
// ALWAYS-INLINE-NEXT:    %5 = arith.addi %1, %2 : i32
// ALWAYS-INLINE-NEXT:    %6 = arith.addi %3, %4 : i32
// ALWAYS-INLINE-NEXT:    %7 = arith.addi %5, %6 : i32
// ALWAYS-INLINE-NEXT:    return %7 : i32
// ALWAYS-INLINE-NEXT:  }

// ALWAYS-INLINE-LABEL: func.func @always_inline_callee
// ALWAYS-INLINE-LABEL: func.func @inline_hint_callee
// ALWAYS-INLINE-LABEL: func.func private @private_callee
// ALWAYS-INLINE-LABEL: func.func @callee
// ALWAYS-INLINE-LABEL: gpu.func @gpu_func_callee

// COM: Ensure a func.func can be inlined in a func.func caller iff the callee is 'alwaysinline'.
// COM: Ensure a gpu.func cannot be inlined in a func.func caller (even if it has the 'alwaysinline' attribute).

// INLINE-LABEL: InlinePass
// INLINE-LABEL:  (S) 3 num-inlined-calls - Number of inlined calls

// INLINE-LABEL: func.func @caller() -> i32 {
// INLINE-NEXT:    %c1_i32 = arith.constant 1 : i32
// INLINE-NEXT:    %c2_i32 = arith.constant 2 : i32
// INLINE-NEXT:    %c3_i32 = arith.constant 3 : i32
// INLINE-NEXT:    %0 = sycl.call() {FunctionName = @callee_, MangledFunctionName = @callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %1 = sycl.call() {FunctionName = @gpu_func_callee_, MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %2 = arith.addi %c1_i32, %c2_i32 : i32
// INLINE-NEXT:    %3 = arith.addi %c3_i32, %0 : i32
// INLINE-NEXT:    %4 = arith.addi %1, %2 : i32
// INLINE-NEXT:    %5 = arith.addi %3, %4 : i32
// INLINE-NEXT:    return %5 : i32
// INLINE-NEXT:  }

// INLINE-LABEL: func.func @always_inline_callee
// INLINE-LABEL: func.func @inline_hint_callee
// INLINE-NOT: func.func private @private_callee
// INLINE-LABEL: func.func @callee
// INLINE-LABEL: gpu.func @gpu_func_callee

gpu.module @module {

func.func @caller() -> i32 {
  %res1 = sycl.call() {FunctionName = @"always_inline_callee_", MangledFunctionName = @always_inline_callee, TypeName = @A} : () -> i32
  %res2 = sycl.call() {FunctionName = @"inline_hint_callee_", MangledFunctionName = @inline_hint_callee, TypeName = @A} : () -> i32
  %res3 = sycl.call() {FunctionName = @"private_callee_", MangledFunctionName = @private_callee, TypeName = @A} : () -> i32
  %res4 = sycl.call() {FunctionName = @"callee_", MangledFunctionName = @callee, TypeName = @A} : () -> i32
  %res5 = sycl.call() {FunctionName = @"gpu_func_callee_", MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
  %res6 = arith.addi %res1, %res2 : i32
  %res7 = arith.addi %res3, %res4 : i32  
  %res8 = arith.addi %res5, %res6 : i32  
  %res9 = arith.addi %res7, %res8 : i32    
  return %res9 : i32
}

func.func @always_inline_callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32 : i32
}

func.func @inline_hint_callee() -> i32 attributes {passthrough = ["inlinehint"]} {
  %c2_i32 = arith.constant 2 : i32
  return %c2_i32 : i32
}

func.func private @private_callee() -> i32 {
  %c3_i32 = arith.constant 3 : i32
  return %c3_i32 : i32
}

func.func @callee() -> i32 {
  %c4_i32 = arith.constant 4 : i32
  return %c4_i32 : i32
}

gpu.func @gpu_func_callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c5_i32 = arith.constant 5 : i32
  gpu.return %c5_i32 : i32
}

}

// -----

// COM: Ensure a func.func can be inlined in a gpu.func caller iff the callee is 'alwaysinline'. 
// ALWAYS-INLINE-LABEL: gpu.func @caller() -> i32 {
// ALWAYS-INLINE-NEXT:    %c1_i32 = arith.constant 1 : i32  
// ALWAYS-INLINE-NEXT:    %0 = sycl.call() {FunctionName = @callee_, MangledFunctionName = @callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %1 = sycl.call() {FunctionName = @gpu_callee_, MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %2 = arith.addi %c1_i32, %0 : i32
// ALWAYS-INLINE-NEXT:    %3 = arith.addi %2, %1 : i32
// ALWAYS-INLINE-NEXT:    gpu.return %3 : i32
// ALWAYS-INLINE-NEXT:  }

// COM: Ensure a func.func can be inlined in a gpu.func caller. 
// INLINE-LABEL: gpu.func @caller() -> i32 {
// INLINE-NEXT:    %c1_i32 = arith.constant 1 : i32  
// INLINE-NEXT:    %0 = sycl.call() {FunctionName = @callee_, MangledFunctionName = @callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %1 = sycl.call() {FunctionName = @gpu_callee_, MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %2 = arith.addi %c1_i32, %0 : i32
// INLINE-NEXT:    %3 = arith.addi %2, %1 : i32
// INLINE-NEXT:    gpu.return %3 : i32
// INLINE-NEXT:  }

gpu.module @module {

gpu.func @caller() -> i32 {
  %res1 = sycl.call() {FunctionName = @"inlinable_callee_", MangledFunctionName = @inlinable_callee, TypeName = @A} : () -> i32
  %res2 = sycl.call() {FunctionName = @"callee_", MangledFunctionName = @callee, TypeName = @A} : () -> i32  
  %res3 = sycl.call() {FunctionName = @"gpu_callee_", MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
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

// -----

// COM: Ensure functions in a SCC are fully inlined (requires multiple inlining iterations). 
// INLINE-LABEL: func.func @callee() -> i32 {
// INLINE-DAG:     %c4_i32 = arith.constant 4 : i32
// INLINE-DAG:     %c5_i32 = arith.constant 5 : i32
// INLINE-DAG:     %c1_i32 = arith.constant 1 : i32
// INLINE-DAG:     %c2_i32 = arith.constant 2 : i32  
// INLINE-DAG:     %c3_i32 = arith.constant 3 : i32
// INLINE-DAG:     %0 = sycl.call() {FunctionName = @callee_, MangledFunctionName = @callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %1 = arith.addi %c3_i32, %0 : i32
// INLINE-NEXT:    %2 = arith.addi %c2_i32, %1 : i32
// INLINE-NEXT:    %3 = arith.addi %c1_i32, %2 : i32
// INLINE-NEXT:    %4 = arith.addi %c5_i32, %3 : i32
// INLINE-NEXT:    %5 = arith.addi %c4_i32, %4 : i32
// INLINE-NEXT:    return %5 : i32
// INLINE-NEXT:  }

// INLINE-NOT: func.func private @inline_hint_callee1
// INLINE-NOT: func.func private @inline_hint_callee2
// INLINE-NOT: func.func private @private_callee
// INLINE-NOT: func.func private @@always_inline_callee

func.func private @inline_hint_callee1() -> i32 attributes {passthrough = ["inlinehint"]} {
  %c_i32 = arith.constant 1 : i32
  %res1 = sycl.call() {FunctionName = @"inline_hint_callee2_", MangledFunctionName = @inline_hint_callee2, TypeName = @A} : () -> i32    
  %res2 = arith.addi %c_i32, %res1 : i32
  return %res2 : i32
}

func.func private @inline_hint_callee2() -> i32 attributes {passthrough = ["inlinehint"]} {
  %c_i32 = arith.constant 2 : i32
  %res1 = sycl.call() {FunctionName = @"private_callee_", MangledFunctionName = @private_callee, TypeName = @A} : () -> i32
  %res2 = arith.addi %c_i32, %res1 : i32
  return %res2 : i32
}

func.func private @private_callee() -> i32 {
  %c_i32 = arith.constant 3 : i32
  %res1 = sycl.call() {FunctionName = @"callee_", MangledFunctionName = @callee, TypeName = @A} : () -> i32
  %res2 = arith.addi %c_i32, %res1 : i32
  return %res2 : i32
}

func.func @callee() -> i32 {
  %c_i32 = arith.constant 4 : i32
  %res1 = sycl.call() {FunctionName = @"always_inline_callee_", MangledFunctionName = @always_inline_callee, TypeName = @A} : () -> i32  
  %res2 = arith.addi %c_i32, %res1 : i32  
  return %res2 : i32
}

func.func private @always_inline_callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c_i32 = arith.constant 5 : i32
  %res1 = sycl.call() {FunctionName = @"inline_hint_callee1_", MangledFunctionName = @inline_hint_callee1, TypeName = @A} : () -> i32    
  %res2 = arith.addi %c_i32, %res1 : i32  
  return %res2 : i32
}
