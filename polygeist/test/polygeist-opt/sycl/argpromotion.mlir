// RUN: polygeist-opt --arg-promotion --split-input-file %s | FileCheck %s

gpu.module @device_func {
  // COM: This function is not a candidate because it doesn't have any argument.
  func.func private @no_args() -> () {
    // CHECK-LABEL: func.func private @no_args
    func.return 
  }

  // COM: This function is not a candidate because the argument doesn't have the expected type.
  func.func private @no_cand_args(%arg0: memref<?xi32>) -> () {
    // CHECK-LABEL: func.func private @no_cand_args
    // CHECK-SAME:    (%arg0: memref<?xi32>) {
    func.return
  }

  // COM: This function is not a candidate because it is not defined.
  func.func private @extern(%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> ()

  // COM: This function is a candidate, check that it is transformed correctly.
  func.func private @callee1(%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> i64 {
    // CHECK-LABEL: func.func private @callee1
    // CHECK-SAME:    (%arg0: memref<?xi32>, %arg1: memref<?xi64>) -> i64 {
    // CHECK-NOT:     {{.*}} = "polygeist.subindex"
    // CHECK:         {{.*}} = affine.load %arg0[0] : memref<?xi32>
    // CHECK-NEXT:    {{.*}} = affine.load %arg1[0] : memref<?xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi32>
    %1 = "polygeist.subindex"(%arg0, %c1) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi64>
    %2 = affine.load %0[0] : memref<?xi32>
    %3 = affine.load %1[0] : memref<?xi64>
    %4 = arith.extsi %2 : i32 to i64
    %5 = arith.addi %3, %4 : i64
    return %5 : i64
  }    

  // COM: This function is not a candidate because one call site uses the argument.
  func.func private @callee2(%arg0: memref<?x!llvm.struct<(i32)>>) {  
    // CHECK-LABEL: func.func private @callee2
    // CHECK-SAME:    (%arg0: memref<?x!llvm.struct<(i32)>>) {      
    // CHECK:         {{.*}} = "polygeist.subindex"
    %c0 = arith.constant 0 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>    
    func.return
  }  

  // COM: Test that the argument of '@callee' is peeled.
  // COM-NEXT: Ensure that the following call sites aren't considered as candidates:
  // COM-NEXT:   - calls to functions with no operands
  // COM-NEXT:   - calls to functions with operands that dont have the expected type 
  // COM-NEXT:   - calls to functions that are't defined
  gpu.func @test1() kernel {
    // CHECK-LABEL: gpu.func @test1() kernel
    // CHECK:         func.call @no_args() : () -> ()
    // CHECK-NEXT:    func.call @extern({{.*}}) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    // CHECK-NEXT:    func.call @no_cand_args({{.*}}) : (memref<?xi32>) -> ()
    // CHECK-NEXT:    %c0 = arith.constant 0 : index
    // CHECK-NEXT:    [[ARG0:%.*]] = "polygeist.subindex"(%cast_3, %c0) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi32>
    // CHECK-NEXT:    %c1 = arith.constant 1 : index
    // CHECK-NEXT:    [[ARG1:%.*]] = "polygeist.subindex"(%cast_3, %c1) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi64>
    // CHECK-NEXT:    {{.*}} = func.call @callee1([[ARG0]], [[ARG1]]) : (memref<?xi32>, memref<?xi64>) -> i64
    // CHECK-NEXT:    gpu.return
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    %alloca_2 = memref.alloca() : memref<1xi32>
    %cast_2 = memref.cast %alloca_2 : memref<1xi32> to memref<?xi32>
    %alloca_3 = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast_3 = memref.cast %alloca_3 : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    func.call @no_args() : () -> ()
    func.call @extern(%cast_1) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()    
    func.call @no_cand_args(%cast_2) : (memref<?xi32>) -> ()
    func.call @callee1(%cast_3) : (memref<?x!llvm.struct<(i32, i64)>>) -> i64
    gpu.return
  }

  // COM: Test that the call is not peeled (the argument is used after the call).
  gpu.func @test2() kernel {
    // CHECK-LABEL: gpu.func @test2() kernel
    // CHECK:         func.call @callee2([[ARG0:%.*]]) : (memref<?x!llvm.struct<(i32)>>) -> ()
    // CHECK-NEXT:    %c0 = arith.constant 0 : index
    // CHECK-NEXT:    {{.*}} = memref.load [[ARG0]][%c0] : memref<?x!llvm.struct<(i32)>>
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i32)>>
    %cast = memref.cast %alloca : memref<1x!llvm.struct<(i32)>> to memref<?x!llvm.struct<(i32)>>
    func.call @callee2(%cast) : (memref<?x!llvm.struct<(i32)>>) -> ()
    %i = arith.constant 0 : index    
    %0 = memref.load %cast[%i] : memref<?x!llvm.struct<(i32)>>
    gpu.return
  }
}

