// RUN: sycl-mlir-opt -split-input-file -inliner="mode=alwaysinline remove-dead-callees=false inline-sycl-method-ops=false" -verify-diagnostics -mlir-pass-statistics %s 2>&1 | FileCheck --check-prefixes=ALWAYS-INLINE,CHECK-ALL %s
// RUN: sycl-mlir-opt -split-input-file -inliner="mode=simple remove-dead-callees=true inline-sycl-method-ops=false" -verify-diagnostics -mlir-pass-statistics %s 2>&1 | FileCheck --check-prefixes=INLINE,CHECK-ALL %s
// RUN: sycl-mlir-opt -split-input-file -inliner="mode=aggressive remove-dead-callees=true inline-sycl-method-ops=false" -verify-diagnostics -mlir-pass-statistics %s 2>&1 | FileCheck --check-prefixes=AGGRESSIVE,CHECK-ALL %s
// RUN: sycl-mlir-opt -split-input-file -inliner="mode=aggressive remove-dead-callees=true inline-sycl-method-ops=true" -verify-diagnostics -mlir-pass-statistics %s 2>&1 | FileCheck --check-prefixes=AGGRESSIVE-METHODOPS,CHECK-ALL %s

// COM: Ensure a func.func can be inlined in a func.func caller iff the callee is 'alwaysinline'.
// COM: Ensure a gpu.func cannot be inlined in a func.func caller (even if it has the 'alwaysinline' attribute).

// ALWAYS-INLINE-LABEL: InlinePass
// ALWAYS-INLINE-LABEL:  (S) 1 num-inlined-calls - Number of inlined calls

// ALWAYS-INLINE-LABEL: func.func @caller() -> i32 {
// ALWAYS-INLINE-NEXT:    %c1_i32 = arith.constant 1 : i32
// ALWAYS-INLINE-NEXT:    %0 = sycl.call @inline_hint_callee_() {MangledFunctionName = @inline_hint_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %1 = sycl.call @private_callee_() {MangledFunctionName = @private_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %2 = sycl.call @small_callee_() {MangledFunctionName = @small_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %3 = sycl.call @callee_() {MangledFunctionName = @callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %4 = sycl.call @gpu_func_callee_() {MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %5 = arith.addi %c1_i32, %0 : i32
// ALWAYS-INLINE-NEXT:    %6 = arith.addi %1, %2 : i32
// ALWAYS-INLINE-NEXT:    %7 = arith.addi %3, %4 : i32
// ALWAYS-INLINE-NEXT:    %8 = arith.addi %5, %6 : i32
// ALWAYS-INLINE-NEXT:    %9 = arith.addi %7, %8 : i32
// ALWAYS-INLINE-NEXT:    return %9 : i32
// ALWAYS-INLINE-NEXT:  }

// ALWAYS-INLINE-LABEL: func.func @always_inline_callee
// ALWAYS-INLINE-LABEL: func.func @inline_hint_callee
// ALWAYS-INLINE-LABEL: func.func private @private_callee
// ALWAYS-INLINE-LABEL: func.func @small_callee
// ALWAYS-INLINE-LABEL: func.func @callee
// ALWAYS-INLINE-LABEL: gpu.func @gpu_func_callee

// COM: Ensure a func.func can be inlined in a func.func caller iff the callee is 'alwaysinline'.
// COM: Ensure a gpu.func cannot be inlined in a func.func caller (even if it has the 'alwaysinline' attribute).

// INLINE-LABEL: InlinePass
// INLINE-LABEL:  (S) 4 num-inlined-calls - Number of inlined calls

// INLINE-LABEL: func.func @caller() -> i32 {
// INLINE-DAG:     %c1_i32 = arith.constant 1 : i32
// INLINE-DAG:     %c2_i32 = arith.constant 2 : i32
// INLINE-DAG:     %c3_i32 = arith.constant 3 : i32
// INLINE-DAG:     %c4_i32 = arith.constant 4 : i32
// INLINE-NEXT:    %0 = sycl.call @callee_() {MangledFunctionName = @callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %1 = sycl.call @gpu_func_callee_() {MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %2 = arith.addi %c1_i32, %c2_i32 : i32
// INLINE-NEXT:    %3 = arith.addi %c3_i32, %c4_i32 : i32
// INLINE-NEXT:    %4 = arith.addi %0, %1 : i32
// INLINE-NEXT:    %5 = arith.addi %2, %3 : i32
// INLINE-NEXT:    %6 = arith.addi %4, %5 : i32
// INLINE-NEXT:    return %6 : i32
// INLINE-NEXT:  }

// INLINE-LABEL: func.func @always_inline_callee
// INLINE-LABEL: func.func @inline_hint_callee
// INLINE-NOT: func.func private @private_callee
// INLINE-LABEL: func.func @small_callee
// INLINE-LABEL: func.func @callee
// INLINE-LABEL: gpu.func @gpu_func_callee

gpu.module @module {

func.func @caller() -> i32 {
  %res1 = sycl.call @always_inline_callee_() {MangledFunctionName = @always_inline_callee, TypeName = @A} : () -> i32
  %res2 = sycl.call @inline_hint_callee_() {MangledFunctionName = @inline_hint_callee, TypeName = @A} : () -> i32
  %res3 = sycl.call @private_callee_() {MangledFunctionName = @private_callee, TypeName = @A} : () -> i32
  %res4 = sycl.call @small_callee_() {MangledFunctionName = @small_callee, TypeName = @A} : () -> i32
  %res5 = sycl.call @callee_() {MangledFunctionName = @callee, TypeName = @A} : () -> i32  
  %res6 = sycl.call @gpu_func_callee_() {MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
  %add1 = arith.addi %res1, %res2 : i32
  %add2 = arith.addi %res3, %res4 : i32  
  %add3 = arith.addi %res5, %res6 : i32  
  %add4 = arith.addi %add1, %add2 : i32    
  %add5 = arith.addi %add3, %add4 : i32      
  return %add5 : i32
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

func.func @small_callee() -> i32 {
  %c4_i32 = arith.constant 4 : i32
  return %c4_i32 : i32
}

func.func @callee() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32  
  %add1 = arith.addi %c1, %c2 : i32
  %add2 = arith.addi %c3, %c4 : i32  
  %add3 = arith.addi %add1, %add2 : i32  
  return %add3 : i32
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
// ALWAYS-INLINE-NEXT:    %0 = sycl.call @callee_() {MangledFunctionName = @callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %1 = sycl.call @gpu_callee_() {MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE-NEXT:    %2 = arith.addi %c1_i32, %0 : i32
// ALWAYS-INLINE-NEXT:    %3 = arith.addi %2, %1 : i32
// ALWAYS-INLINE-NEXT:    gpu.return %3 : i32
// ALWAYS-INLINE-NEXT:  }

// COM: Ensure a func.func can be inlined in a gpu.func caller. 
// INLINE-LABEL: gpu.func @caller() -> i32 {
// INLINE-NEXT:    %c1_i32 = arith.constant 1 : i32  
// INLINE-NEXT:    %0 = sycl.call @callee_() {MangledFunctionName = @callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %1 = sycl.call @gpu_callee_() {MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
// INLINE-NEXT:    %2 = arith.addi %c1_i32, %0 : i32
// INLINE-NEXT:    %3 = arith.addi %2, %1 : i32
// INLINE-NEXT:    gpu.return %3 : i32
// INLINE-NEXT:  }

gpu.module @module {

gpu.func @caller() -> i32 {
  %res1 = sycl.call @inlinable_callee_() {MangledFunctionName = @inlinable_callee, TypeName = @A} : () -> i32
  %res2 = sycl.call @callee_() {MangledFunctionName = @callee, TypeName = @A} : () -> i32  
  %res3 = sycl.call @gpu_callee_() {MangledFunctionName = @gpu_func_callee, TypeName = @A} : () -> i32
  %res4 = arith.addi %res1, %res2 : i32
  %res5 = arith.addi %res4, %res3 : i32  
  gpu.return %res5 : i32
}

func.func @inlinable_callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32 : i32
}

func.func @callee() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32  
  %add1 = arith.addi %c1, %c2 : i32
  %add2 = arith.addi %c3, %c4 : i32  
  %add3 = arith.addi %add1, %add2 : i32  
  return %add3 : i32
}

gpu.func @gpu_func_callee() -> i32 attributes {passthrough = ["alwaysinline"]} {
  %c3_i32 = arith.constant 3 : i32
  gpu.return %c3_i32 : i32
}

}

// -----

// AGGRESSIVE-METHODOPS-NOT: func.func private @inline_hint_callee
// AGGRESSIVE-METHODOPS-NOT: func.func private @private_callee
// AGGRESSIVE-METHODOPS-NOT: func.func private @get

// AGGRESSIVE-NOT: func.func private @inline_hint_callee
// AGGRESSIVE-NOT: func.func private @private_callee

// COM: Ensure functions in a SCC are fully inlined (requires multiple inlining iterations).
// CHECK-ALL-LABEL:   func.func @main(
// CHECK-ALL-SAME:                      %[[VAL_0:.*]]: memref<?x!sycl_id_1_>) -> (i32, i64) {

// ALWAYS-INLINE:           %[[VAL_1:.*]] = arith.constant 1 : i32
// ALWAYS-INLINE:           %[[VAL_2:.*]] = sycl.call @inline_hint_callee_() {MangledFunctionName = @inline_hint_callee, TypeName = @A} : () -> i32
// ALWAYS-INLINE:           call @foo(%[[VAL_0]], %[[VAL_1]]) : (memref<?x!sycl_id_1_>, i32) -> ()
// ALWAYS-INLINE:           %[[VAL_3:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_1]]] {ArgumentTypes = [memref<?x!sycl_id_1_, 4>, i32], FunctionName = @get, MangledFunctionName = @get, TypeName = @id} : (memref<?x!sycl_id_1_>, i32) -> i64
// ALWAYS-INLINE:           return %[[VAL_2]], %[[VAL_3]] : i32, i64
// ALWAYS-INLINE:         }

// INLINE-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// INLINE-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i32
// INLINE-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// INLINE:           %[[VAL_4:.*]] = sycl.call @main_() {MangledFunctionName = @main, TypeName = @A} : () -> i32
// INLINE:           %[[VAL_5:.*]] = arith.addi %[[VAL_3]], %[[VAL_4]] : i32
// INLINE:           %[[VAL_6:.*]] = arith.addi %[[VAL_2]], %[[VAL_5]] : i32
// INLINE:           call @foo(%[[VAL_0]], %[[VAL_1]]) : (memref<?x!sycl_id_1_>, i32) -> ()
// INLINE:           %[[VAL_7:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_1]]] {ArgumentTypes = [memref<?x!sycl_id_1_, 4>, i32], FunctionName = @get, MangledFunctionName = @get, TypeName = @id} : (memref<?x!sycl_id_1_>, i32) -> i64
// INLINE:           return %[[VAL_6]], %[[VAL_7]] : i32, i64
// INLINE:         }

// INLINE-NOT: func.func private @inline_hint_callee
// INLINE-NOT: func.func private @private_callee

// AGGRESSIVE-METHODOPS-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// AGGRESSIVE-METHODOPS-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i32
// AGGRESSIVE-METHODOPS-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// AGGRESSIVE-METHODOPS:           %[[VAL_4:.*]] = sycl.call @main_() {MangledFunctionName = @main, TypeName = @A} : () -> i32
// AGGRESSIVE-METHODOPS:           %[[VAL_5:.*]] = arith.addi %[[VAL_3]], %[[VAL_4]] : i32
// AGGRESSIVE-METHODOPS:           %[[VAL_6:.*]] = arith.addi %[[VAL_2]], %[[VAL_5]] : i32
// AGGRESSIVE-METHODOPS:           call @foo(%[[VAL_0]], %[[VAL_1]]) : (memref<?x!sycl_id_1_>, i32) -> ()
// AGGRESSIVE-METHODOPS:           %[[VAL_7:.*]] = memref.memory_space_cast %[[VAL_0]] : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
// AGGRESSIVE-METHODOPS:           %[[VAL_8:.*]] = arith.constant 2 : i64
// AGGRESSIVE-METHODOPS:           return %[[VAL_6]], %[[VAL_8]] : i32, i64
// AGGRESSIVE-METHODOPS:         }

// AGGRESSIVE:           %[[VAL_1:.*]] = arith.constant 1 : i32
// AGGRESSIVE:           %[[VAL_2:.*]] = arith.constant 1 : i32
// AGGRESSIVE:           %[[VAL_3:.*]] = arith.constant 2 : i32
// AGGRESSIVE:           %[[VAL_4:.*]] = sycl.call @main_() {MangledFunctionName = @main, TypeName = @A} : () -> i32
// AGGRESSIVE:           %[[VAL_5:.*]] = arith.addi %[[VAL_3]], %[[VAL_4]] : i32
// AGGRESSIVE:           %[[VAL_6:.*]] = arith.addi %[[VAL_2]], %[[VAL_5]] : i32
// AGGRESSIVE:           call @foo(%[[VAL_0]], %[[VAL_1]]) : (memref<?x!sycl_id_1_>, i32) -> ()
// AGGRESSIVE:           %[[VAL_7:.*]] = sycl.id.get %[[VAL_0]]{{\[}}%[[VAL_1]]] {ArgumentTypes = [memref<?x!sycl_id_1_, 4>, i32], FunctionName = @get, MangledFunctionName = @get, TypeName = @id} : (memref<?x!sycl_id_1_>, i32) -> i64
// AGGRESSIVE:           return %[[VAL_6]], %[[VAL_7]] : i32, i64
// AGGRESSIVE:         }

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>

func.func private @inline_hint_callee() -> i32 attributes {passthrough = ["inlinehint"]} {
  %c_i32 = arith.constant 1 : i32
  %res1 = sycl.call @private_callee_() {MangledFunctionName = @private_callee, TypeName = @A} : () -> i32
  %res2 = arith.addi %c_i32, %res1 : i32
  return %res2 : i32
}

func.func private @private_callee() -> i32 {
  %c_i32 = arith.constant 2 : i32
  %res1 = sycl.call @main_() {MangledFunctionName = @main, TypeName = @A} : () -> i32
  %res2 = arith.addi %c_i32, %res1 : i32
  return %res2 : i32
}

func.func private @get(%id: memref<?x!sycl_id_1_, 4>, %i: i32) -> i64 {
  %c2_i64 = arith.constant 2 : i64
  return %c2_i64 : i64
}

func.func private @foo(%id: memref<?x!sycl_id_1_>, %i: i32)

func.func private @id(%id: memref<?x!sycl_id_1_>, %i: i32) attributes {passthrough = ["alwaysinline"]} {
  func.call @foo(%id, %i) : (memref<?x!sycl_id_1_>, i32) -> ()
  return
}

func.func @main(%id: memref<?x!sycl_id_1_>) -> (i32, i64) {
  %c1_i32 = arith.constant 1 : i32
  %res1 = sycl.call @inline_hint_callee_() {MangledFunctionName = @inline_hint_callee, TypeName = @A} : () -> i32    
  sycl.constructor @id(%id, %c1_i32) {MangledFunctionName = @id} : (memref<?x!sycl_id_1_>, i32)
  %res2 = sycl.id.get %id[%c1_i32] {ArgumentTypes = [memref<?x!sycl_id_1_, 4>, i32], FunctionName = @get, MangledFunctionName = @get, TypeName = @id} : (memref<?x!sycl_id_1_>, i32) -> i64
  return %res1, %res2 : i32, i64
}
