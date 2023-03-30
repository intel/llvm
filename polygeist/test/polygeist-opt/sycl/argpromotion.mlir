// RUN: polygeist-opt --arg-promotion --split-input-file %s | FileCheck %s

#map = affine_map<(s0) -> (s0 - 1)>  
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

  // COM: This function is not a candidate because one call site uses the argument after the call.
  func.func private @callee2(%arg0: memref<?x!llvm.struct<(i32, i64)>>) {  
    // CHECK-LABEL: func.func private @callee2
    // CHECK-SAME:    (%arg0: memref<?x!llvm.struct<(i32, i64)>>) {      
    // CHECK:         {{.*}} = "polygeist.subindex"
    %c0 = arith.constant 0 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi32>    
    func.return
  }  

  // COM: This function is a candidate, check that it is transformed correctly.
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

  // COM: Should not peel the argument because it is a multidimentional memref.
  func.func private @callee5(%arg0: memref<?x1x!llvm.struct<(i32)>>) {
    // CHECK-LABEL: func.func private @callee5
    // CHECK-SAME:    (%arg0: memref<?x1x!llvm.struct<(i32)>>) {
    func.return
  }

  // COM: Should not peel the argument because it is memref with non-identity layout.
  func.func private @callee6(%arg0: memref<?x!llvm.struct<(i32)>, #map>) {
    // CHECK-LABEL: func.func private @callee6
    // CHECK-SAME:    (%arg0: memref<?x!llvm.struct<(i32)>, #map>) {
    func.return
  }

  // COM: Should not peel the argument because it is memref with an invalid element type.
  func.func private @callee7(%arg0: memref<?x!llvm.struct<(struct<(i32)>)>>) {
    // CHECK-LABEL: func.func private @callee7
    // CHECK-SAME:    (%arg0: memref<?x!llvm.struct<(struct<(i32)>)>>) {
    func.return
  }

  // COM: This function is a candidate, check that it is transformed correctly.
  func.func @callee8(%arg0: memref<?x!llvm.struct<(i32, i64)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    // CHECK-LABEL: func.func @callee8
    // CHECK-SAME:    (%arg0: memref<?xi32> {llvm.noalias}, %arg1: memref<?xi64> {llvm.noalias})
    // CHECK-SAME:    attributes {llvm.linkage = #llvm.linkage<internal>} {
    func.return
  }

  // COM: Test that a call with a single peelable argument is peeled correctly.
  // COM-NEXT: Ensure that the following call sites aren't considered as candidates:
  // COM-NEXT:   - calls to functions with no operands
  // COM-NEXT:   - calls to functions with operands that dont have the expected type
  gpu.func @test1() kernel {
    // CHECK-LABEL: gpu.func @test1() kernel
    // CHECK:         func.call @no_args() : () -> ()
    // CHECK-NEXT:    func.call @no_cand_args({{.*}}) : (memref<?xi32>) -> ()
    // CHECK-NEXT:    %c0 = arith.constant 0 : index
    // CHECK-NEXT:    [[ARG0:%.*]] = "polygeist.subindex"(%cast, %c0) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi32>
    // CHECK-NEXT:    %c1 = arith.constant 1 : index
    // CHECK-NEXT:    [[ARG1:%.*]] = "polygeist.subindex"(%cast, %c1) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi64>
    // CHECK-NEXT:    {{.*}} = func.call @callee1([[ARG0]], [[ARG1]]) : (memref<?xi32>, memref<?xi64>) -> i64
    // CHECK-NEXT:    gpu.return
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    %alloca_2 = memref.alloca() : memref<1xi32>
    %cast_2 = memref.cast %alloca_2 : memref<1xi32> to memref<?xi32>
    func.call @no_args() : () -> ()
    func.call @no_cand_args(%cast_2) : (memref<?xi32>) -> ()
    func.call @callee1(%cast_1) : (memref<?x!llvm.struct<(i32, i64)>>) -> i64
    gpu.return
  }

  // COM: Test that a call with a peelable argument that is used after the call is not modified.
  gpu.func @test2() kernel {
    // CHECK-LABEL: gpu.func @test2() kernel
    // CHECK:         func.call @callee2([[ARG0:%.*]]) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast = memref.cast %alloca : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    func.call @callee2(%cast) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    %i = arith.constant 0 : index
    %0 = memref.load %cast[%i] : memref<?x!llvm.struct<(i32, i64)>>
    gpu.return
  }

  // COM: Test that the a call to an externally defined function is not modified.
  gpu.func @test3() kernel {  
    // CHECK-LABEL: gpu.func @test3() kernel
    // CHECK:         func.call @extern({{.*}}) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>    
    func.call @extern(%cast_1) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    gpu.return    
  }

  // COM: Test that multiple peelable arguments are peeled correctly.
  gpu.func @test4() kernel {
    // CHECK-LABEL: gpu.func @test4() kernel
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
  gpu.func @test5() kernel {
    // CHECK-LABEL: gpu.func @test5() kernel
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

  // COM: Test that an argument that is a multidimentionsal memref is not peeled.
  gpu.func @test6() kernel {
    // CHECK-LABEL: gpu.func @test6() kernel
    // CHECK-NOT:     {{.*}} = "polygeist.subindex"
    // CHECK:         func.call @callee5({{.*}}) : (memref<?x1x!llvm.struct<(i32)>>) -> ()
    %alloca_1 = memref.alloca() : memref<1x1x!llvm.struct<(i32)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x1x!llvm.struct<(i32)>> to memref<?x1x!llvm.struct<(i32)>>
    func.call @callee5(%cast_1) : (memref<?x1x!llvm.struct<(i32)>>) -> ()
    gpu.return
  }

  // COM: Test that an argument that is a memref with non-id layouit is not peeled.
  gpu.func @test7() kernel {
    // CHECK-LABEL: gpu.func @test7() kernel
    %alloca_1 = memref.alloca() : memref<8x!llvm.struct<(i32)>, #map>
    %cast_1 = memref.cast %alloca_1 : memref<8x!llvm.struct<(i32)>, #map> to memref<?x!llvm.struct<(i32)>, #map>
    func.call @callee6(%cast_1) : (memref<?x!llvm.struct<(i32)>, #map>) -> ()
    gpu.return
  }

  // COM: Test that an argument that is a memref with element type 'struct<(struct<(...)>)>' is not peeled.
  gpu.func @test8() kernel {
    // CHECK-LABEL: gpu.func @test8() kernel
    %alloca_1 = memref.alloca() : memref<3x!llvm.struct<(struct<(i32)>)>>
    %cast_1 = memref.cast %alloca_1 : memref<3x!llvm.struct<(struct<(i32)>)>> to memref<?x!llvm.struct<(struct<(i32)>)>>
    func.call @callee7(%cast_1) : (memref<?x!llvm.struct<(struct<(i32)>)>>) -> ()
    gpu.return
  }

  // COM: Test that peelable arguments can be peeled if the end function (callee1) is called from more than one call site:
  // COM:   test9 -> callee1_wrapper -> callee1
  // COM:   test1 -> callee1
  gpu.func @test9() kernel {
    // CHECK-LABEL: gpu.func @test9() kernel
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

  // COM: Test that the a call to a linkonce_odr function is modified.
  gpu.func @test10() kernel {  
    // CHECK-LABEL: gpu.func @test10() kernel
    // CHECK:         %c0 = arith.constant 0 : index
    // CHECK-NEXT:    [[ARG0:%.*]] = "polygeist.subindex"(%cast, %c0) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi32>
    // CHECK-NEXT:    %c1 = arith.constant 1 : index
    // CHECK-NEXT:    [[ARG1:%.*]] = "polygeist.subindex"(%cast, %c1) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi64>
    // CHECK-NEXT:    func.call @callee8([[ARG0]], [[ARG1]]) : (memref<?xi32>, memref<?xi64>) -> ()
    // CHECK-NEXT:    gpu.return
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>    
    func.call @callee8(%cast_1) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    gpu.return    
  }
}
