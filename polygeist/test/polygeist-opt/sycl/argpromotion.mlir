// RUN: polygeist-opt --arg-promotion --split-input-file %s | FileCheck %s

gpu.module @device_func {
  // COM: This function is a candidate, check that it is transformed correctly.
  func.func private @callee1(%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> i64 {
    // CHECK-LABEL: func.func private @callee1
    // CHECK-SAME:    (%arg0: memref<?xi32> {llvm.noalias}, %arg1: memref<?xi64> {llvm.noalias}) -> i64 {
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

  // COM: Test that a call with a single peelable argument is peeled correctly.
  gpu.func @test1() kernel {
    // CHECK-LABEL: gpu.func @test1() kernel
    // CHECK:         %c0 = arith.constant 0 : index
    // CHECK-NEXT:    [[ARG0:%.*]] = "polygeist.subindex"(%cast, %c0) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi32>
    // CHECK-NEXT:    %c1 = arith.constant 1 : index
    // CHECK-NEXT:    [[ARG1:%.*]] = "polygeist.subindex"(%cast, %c1) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi64>
    // CHECK-NEXT:    {{.*}} = func.call @callee1([[ARG0]], [[ARG1]]) : (memref<?xi32>, memref<?xi64>) -> i64
    // CHECK-NEXT:    gpu.return
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    %alloca_2 = memref.alloca() : memref<1xi32>
    %cast_2 = memref.cast %alloca_2 : memref<1xi32> to memref<?xi32>
    func.call @callee1(%cast_1) : (memref<?x!llvm.struct<(i32, i64)>>) -> i64
    gpu.return
  }

  // COM: Test that peelable arguments can be peeled if the end function (callee1) is called from more than one call site:
  // COM:   test2 -> callee1_wrapper -> callee1
  // COM:   test1 -> callee1
  gpu.func @test2() kernel {
    // CHECK-LABEL: gpu.func @test2() kernel
    // CHECK:         {{.*}} = func.call @callee1_wrapper({{.*}}) : (memref<?x!llvm.struct<(i32, i64)>>) -> i64
    %alloca_1 = memref.alloca() : memref<3x!llvm.struct<(i32, i64)>>
    %cast_1 = memref.cast %alloca_1 : memref<3x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    %0 = func.call @callee1_wrapper(%cast_1) : (memref<?x!llvm.struct<(i32, i64)>>) -> (i64)
    gpu.return
  }

  // COM: Ensure that the peelable argument is peeled if there is a non-aliased instruction after the call.
  func.func private @callee1_wrapper(%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> i64 {
    // CHECK-LABEL: func.func private @callee1_wrapper
    // CHECK-SAME:    (%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> i64
    // CHECK:         %c0 = arith.constant 0 : index    
    // CHECK-NEXT:    [[ARG0:%.*]] = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi32>
    // CHECK-NEXT:    %c1 = arith.constant 1 : index
    // CHECK-NEXT:    [[ARG1:%.*]] = "polygeist.subindex"(%arg0, %c1) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi64>
    // CHECK-NEXT:    {{.*}} = call @callee1([[ARG0]], [[ARG1]]) : (memref<?xi32>, memref<?xi64>) -> i64
    %alloca = memref.alloca() : memref<i64>        
    %0 = func.call @callee1(%arg0) : (memref<?x!llvm.struct<(i32, i64)>>) -> i64
    %1 = memref.load %alloca[] : memref<i64>
    %add = arith.addi %0, %1 : i64
    func.return %add : i64
  }

  // COM: Test that multiple peelable arguments are peeled correctly.
  func.func private @callee3(%arg0: memref<?x!llvm.struct<(i32)>>, %arg1: memref<?x!llvm.struct<(f32)>>) {
    // CHECK-LABEL: func.func private @callee3
    // CHECK-SAME:    (%arg0: memref<?xi32>, %arg1: memref<?xf32>) {
    // CHECK-NOT:     {{.*}} = "polygeist.subindex"
    // CHECK:         {{.*}} = affine.load %arg0[0] : memref<?xi32>
    // CHECK-NEXT:    {{.*}} = affine.load %arg1[0] : memref<?xf32>
    %c0 = arith.constant 0 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
    %1 = "polygeist.subindex"(%arg1, %c0) : (memref<?x!llvm.struct<(f32)>>, index) -> memref<?xf32>
    %2 = affine.load %0[0] : memref<?xi32>
    %3 = affine.load %1[0] : memref<?xf32>
    func.return
  }
  gpu.func @test3() kernel {
    // CHECK-LABEL: gpu.func @test3() kernel
    // CHECK:         [[C0:%.*]] = arith.constant 0 : index
    // CHECK-NEXT:    [[ARG0:%.*]] = "polygeist.subindex"({{.*}}, [[C0]]) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
    // CHECK-NEXT:    [[C0_1:%.*]] = arith.constant 0 : index
    // CHECK-NEXT:    [[ARG1:%.*]] = "polygeist.subindex"({{.*}}, [[C0_1]]) : (memref<?x!llvm.struct<(f32)>>, index) -> memref<?xf32>
    // CHECK-NEXT:    func.call @callee3([[ARG0]], [[ARG1]]) : (memref<?xi32>, memref<?xf32>) -> ()
    // CHECK-NEXT:    gpu.return
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i32)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x!llvm.struct<(i32)>> to memref<?x!llvm.struct<(i32)>>
    %alloca_2 = memref.alloca() : memref<1x!llvm.struct<(f32)>>
    %cast_2 = memref.cast %alloca_2 : memref<1x!llvm.struct<(f32)>> to memref<?x!llvm.struct<(f32)>>
    func.call @callee3(%cast_1, %cast_2) : (memref<?x!llvm.struct<(i32)>>, memref<?x!llvm.struct<(f32)>>) -> ()
    gpu.return
  }

  // COM: Test that a peelable argument can be peeled when another argument with the expected type 
  // COM: cannot be peeled (because used in the callee by an invalid instruction).
  // COM: The first argument in this function can be peeled but the second cannot.
  func.func private @callee4(%arg0: memref<?x!llvm.struct<(i32)>>, %arg1: memref<?x!llvm.struct<(f32)>>) {
    // CHECK-LABEL: func.func private @callee4
    // CHECK-SAME:    (%arg0: memref<?xi32> {llvm.noalias}, %arg1: memref<?x!llvm.struct<(f32)>>) {
    // CHECK-NOT:     {{.*}} = "polygeist.subindex"
    // CHECK:         {{.*}} = sycl.addrspacecast %arg1 : memref<?x!llvm.struct<(f32)>> to memref<?x!llvm.struct<(f32)>, 4>
    // CHECK-NEXT:    {{.*}} = affine.load %arg0[0] : memref<?xi32>
    %c0 = arith.constant 0 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
    %1 = sycl.addrspacecast %arg1 : memref<?x!llvm.struct<(f32)>> to memref<?x!llvm.struct<(f32)>, 4>
    %2 = affine.load %0[0] : memref<?xi32>
    func.return
  }  
  gpu.func @test4() kernel {
    // CHECK-LABEL: gpu.func @test4() kernel
    // CHECK:         [[C0:%.*]] = arith.constant 0 : index
    // CHECK-NEXT:    [[ARG0:%.*]] = "polygeist.subindex"({{.*}}, [[C0]]) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
    // CHECK-NEXT:    func.call @callee4([[ARG0]], {{.*}}) : (memref<?xi32>, memref<?x!llvm.struct<(f32)>>) -> ()    
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i32)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x!llvm.struct<(i32)>> to memref<?x!llvm.struct<(i32)>>
    %alloca_2 = memref.alloca() : memref<1x!llvm.struct<(f32)>>
    %cast_2 = memref.cast %alloca_2 : memref<1x!llvm.struct<(f32)>> to memref<?x!llvm.struct<(f32)>>
    func.call @callee4(%cast_1, %cast_2) : (memref<?x!llvm.struct<(i32)>>, memref<?x!llvm.struct<(f32)>>) -> ()
    gpu.return
  }
}
