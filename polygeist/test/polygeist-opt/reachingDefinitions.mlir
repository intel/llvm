// RUN: polygeist-opt -split-input-file -test-reaching-definition %s 2>&1 | FileCheck %s

!sycl_id_1 = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1 = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2 = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_accessor_1_f32_rw_dev = !sycl.accessor<[1, f32, read_write, device], (!sycl.accessor_impl_device<[1], (!sycl_id_1, !sycl_range_1, !sycl_range_1)>, !llvm.struct<(memref<?xf32, 1>)>)>

// COM: Test that the correct definition reaches a load.
// CHECK-LABEL: test_tag: test1_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test1_store1
// CHECK-NEXT: - pMods: <none>
// CHECK-LABEL: test_tag: test1_load2
// CHECK: operand #0
// CHECK-NEXT: - mods: test1_store2
// CHECK-NEXT: - pMods: <none>
func.func @test1(%val: i32, %idx: index) {
  %alloca = memref.alloca() : memref<1xi32>
  %alloca_0 = memref.alloca() : memref<1xi32>
  memref.store %val, %alloca[%idx] {tag_name = "test1_store1"} : memref<1xi32>
  %1 = memref.load %alloca[%idx] {tag = "test1_load1"} : memref<1xi32>
  memref.store %1, %alloca_0[%idx] {tag_name = "test1_store2"} : memref<1xi32>
  %2 = memref.load %alloca_0[%idx] {tag = "test1_load2"} : memref<1xi32>
  return
}

// COM: Test that the correct definition reaches a load in the presence of a cast.
// CHECK-LABEL: test_tag: test2_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test2_store1
// CHECK-NEXT: - pMods: <none>
func.func @test2(%val: i32, %idx: index) {
  %alloca = memref.alloca() : memref<1xi32>
  %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
  memref.store %val, %cast[%idx] {tag_name = "test2_store1"} : memref<?xi32>
  %1 = memref.load %alloca[%idx] {tag = "test2_load1"} : memref<1xi32>
  return
}

// COM: Test that a (must) aliased definition kills a previous definition.
// CHECK-LABEL: test_tag: test3_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test3_store2
// CHECK-NEXT: - pMods: <none>
func.func @test3(%val : i32, %idx: index) {
  %alloca = memref.alloca()  : memref<1xi32>
  %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
  %addrspace_cast = memref.memory_space_cast %cast : memref<?xi32> to memref<?xi32, 4>
  memref.store %val, %addrspace_cast[%idx] {tag_name = "test3_store1"} : memref<?xi32, 4>
  // The following store kills the previous one because %alloca and %cast are definetely aliased.
  memref.store %val, %alloca[%idx] {tag_name = "test3_store2"} : memref<1xi32>  
  %1 = memref.load %cast[%idx] {tag = "test3_load1"} : memref<?xi32>
  return
}

// COM: Test that a definition and a potential definition both reach a load.
// CHECK-LABEL: test_tag: test4_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: <initial> test4_store2
// CHECK-NEXT: - pMods: test4_store1
func.func @test4(%cond: i1, %arg1: memref<i32>, %arg2: memref<i32>) {
  scf.if %cond {
    %c0 = arith.constant 0 : i32
    memref.store %c0, %arg2[] {tag_name = "test4_store1"}: memref<i32>
  } else {
    %c1 = arith.constant 1 : i32
    memref.store %c1, %arg1[] {tag_name = "test4_store2"}: memref<i32>    
  }
  %1 = memref.load %arg1[] {tag = "test4_load1"} : memref<i32>
  return
}

// COM: Test that potential definitions do not kill previous definitions and 
//      are joined correctly.
// CHECK-LABEL: test_tag: test5_load1
// CHECK: operand #0
// CHECK-NEXT: - mods:
// CHECK-DAG:          test5_store1
// CHECK-DAG:          test5_store3
// CHECK-NEXT: - pMods:
// CHECK-DAG:          test5_store2
// CHECK-DAG:          test5_store4
func.func @test5(%cond: i1, %arg1: memref<i32>, %arg2: memref<i32>) {
  scf.if %cond {
    %c0 = arith.constant 0 : i32
    memref.store %c0, %arg1[] {tag_name = "test5_store1"}: memref<i32>
    memref.store %c0, %arg2[] {tag_name = "test5_store2"}: memref<i32>
  } else {
    %c1 = arith.constant 1 : i32
    memref.store %c1, %arg1[] {tag_name = "test5_store3"}: memref<i32>    
    memref.store %c1, %arg2[] {tag_name = "test5_store4"} : memref<i32>
  }
  %1 = memref.load %arg1[] {tag = "test5_load1"} : memref<i32>
  return
}

// COM: Test that a definition kills a previous potential definition.
// CHECK-LABEL: test_tag: test6_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test6_store2
// CHECK-NEXT: - pMods: <none>
func.func @test6(%arg1: memref<i32>, %arg2: memref<i32>) {
  %c0 = arith.constant 0 : i32
  memref.store %c0, %arg2[] {tag_name = "test6_store1"}: memref<i32>
  memref.store %c0, %arg1[] {tag_name = "test6_store2"}: memref<i32> 
  %1 = memref.load %arg1[] {tag = "test6_load1"} : memref<i32>
  return
}

// COM: Test that a deallocation kills a previous potential definition.
// CHECK-LABEL: test_tag: test7_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: <initial>
// CHECK-NEXT: - pMods: test7_store1
// CHECK-LABEL: test_tag: test7_load2
// CHECK: operand #0
// CHECK-NEXT: - mods: <initial>
// CHECK-NEXT: - pMods: <none>
func.func @test7(%arg1: memref<i32>, %arg2: memref<i32>) {
  %c0 = arith.constant 0 : i32
  memref.store %c0, %arg2[] {tag_name = "test7_store1"} : memref<i32>
  %1 = memref.load %arg1[] {tag = "test7_load1"} : memref<i32>  
  memref.dealloc %arg2 : memref<i32>
  %2 = memref.load %arg1[] {tag = "test7_load2"} : memref<i32>
  return
}

// COM: Test that a deallocation kills a (must) aliased definition.
// CHECK-LABEL: test_tag: test8_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test8_store1
// CHECK-NEXT: - pMods: <none>
// CHECK-LABEL: test_tag: test8_load2
// CHECK: operand #0
// CHECK-NEXT: - mods: <none>
// CHECK-NEXT: - pMods: <none>
// CHECK-LABEL: test_tag: test8_load3
// CHECK: operand #0
// CHECK-NEXT: - mods: <none>
// CHECK-NEXT: - pMods: <none>
func.func @test8(%val: i32, %idx : index) {
  %alloc = memref.alloc() : memref<1xi32>
  %cast = memref.cast %alloc : memref<1xi32> to memref<?xi32>
  memref.store %val, %cast[%idx] {tag_name = "test8_store1"} : memref<?xi32>
  %1 = memref.load %alloc[%idx] {tag = "test8_load1"} : memref<1xi32>
  memref.dealloc %alloc {tag_name = "test8_dealloc1"} : memref<1xi32>
  %2 = memref.load %alloc[%idx] {tag = "test8_load2"} : memref<1xi32>
  %3 = memref.load %cast[%idx] {tag = "test8_load3"} : memref<?xi32>
  return
}

// COM: Ensure the initial definition reaches a load.
// CHECK-LABEL: test_tag: test9_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: <initial>
// CHECK-NEXT: - pMods: <none>
func.func @test9(%arg1: memref<i32>) {
  %1 = memref.load %arg1[] {tag = "test9_load1"} : memref<i32>
  return
}

// COM: Test that an operation with unknown side effects kills a previous definition.
// CHECK-LABEL: test_tag: test10_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test10_store1
// CHECK-NEXT: - pMods: <none>
// CHECK-LABEL: test_tag: test10_load2
// CHECK: operand #0
// CHECK-NEXT: - mods: <unknown>
// CHECK-NEXT: - pMods: <unknown>
func.func private @foo(%arg0: memref<i32>) -> ()

func.func @test10(%val: i32) {
  %alloca = memref.alloca() : memref<i32>
  memref.store %val, %alloca[] {tag_name = "test10_store1"} : memref<i32>
  %1 = memref.load %alloca[] {tag = "test10_load1"} : memref<i32>  
  func.call @foo(%alloca) : (memref<i32>) -> ()
  %3 = memref.load %alloca[] {tag = "test10_load2"} : memref<i32>
  return
}

// COM: Test effects of a deallocation in the presence of control flow.
// CHECK-LABEL: test_tag: test11_load1
// CHECK: operand #0
// CHECK-NEXT: - mods:
// CHECK-DAG:          test11_store1
// CHECK-DAG:          test11_store2
// CHECK-NEXT: - pMods: <none>
// CHECK-LABEL: test_tag: test11_load2
// CHECK: operand #0
// CHECK-NEXT: - mods: <none>
// CHECK-NEXT: - pMods: <none>
func.func @test11(%cond: i1, %val: i32, %arg1: memref<i32>, %arg2: memref<i32>) {
  memref.store %val, %arg1[] {tag_name = "test11_store1"} : memref<i32>
  scf.if %cond {
    %c0 = arith.constant 0 : i32    
    memref.store %c0, %arg1[] {tag_name = "test11_store2"} : memref<i32>
  } else {
    %c1 = arith.constant 1 : i32      
    memref.store %c1, %arg2[] {tag_name = "test11_store3"} : memref<i32>
  }
  memref.dealloc %arg2 {tag_name = "test11_dealloc1"} : memref<i32>
  %1 = memref.load %arg1[] {tag = "test11_load1"} : memref<i32>
  %2 = memref.load %arg2[] {tag = "test11_load2"} : memref<i32>
  return
}

// COM: Test load after load in the presence of control flow.
// CHECK-LABEL: test_tag: test12_load1:
// CHECK-NEXT:  operand #0
// CHECK-NEXT:  - mods: <initial> test12_store1
// CHECK-NEXT:  - pMods: <none>
// CHECK-LABEL: test_tag: test12_load2:
// CHECK-NEXT:  operand #0
// CHECK-NEXT:  - mods: <initial>
// CHECK-NEXT:  - pMods: test12_store1
func.func @test12(%cond: i1, %arg1: memref<i32>, %arg2: memref<i32>) {
  scf.if %cond {
    %val = arith.constant 0 : i32
    memref.store %val, %arg1[] {tag_name = "test12_store1"}: memref<i32>
    scf.yield
  }
  else {
    scf.yield
  }
  %1 = memref.load %arg1[] {tag = "test12_load1"} : memref<i32>
  %2 = memref.load %arg2[] {tag = "test12_load2"} : memref<i32>
  return
}

// COM: Test control flow if without else.
// CHECK-LABEL: test_tag: test13_load1:
// CHECK-NEXT:  operand #0
// CHECK-NEXT:  - mods:
// CHECK-DAG:          test13_store1
// CHECK-DAG:          test13_store2
// CHECK-NEXT:  - pMods: <none>
func.func @test13(%cond: i1, %arg1: memref<i32>) {
  %val = arith.constant 0 : i32
  memref.store %val, %arg1[] {tag_name = "test13_store1"}: memref<i32>
  scf.if %cond {
    memref.store %val, %arg1[] {tag_name = "test13_store2"}: memref<i32>
    scf.yield
  }
  %1 = memref.load %arg1[] {tag = "test13_load1"} : memref<i32>
  return
}

// COM: Test that a definition created by a sycl.constructor reaches a load.
// CHECK-LABEL: test_tag: test14_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test14_store1
// CHECK-NEXT: - pMods: <none>
func.func @test14(%val: i64) {
  %alloca = memref.alloca() : memref<1x!sycl_id_1>
  %addrspace_cast = memref.memory_space_cast %alloca : memref<1x!sycl_id_1> to memref<1x!sycl_id_1, 4>
  sycl.constructor @id(%addrspace_cast, %val) {MangledFunctionName = @constr, tag_name = "test14_store1"} : (memref<1x!sycl_id_1, 4>, i64)
  %2 = affine.load %alloca[0] {tag = "test14_load1"} : memref<1x!sycl_id_1>
  return
}

// COM: Test that a definition reaches a sycl.accessor.subscript operation.
// CHECK-LABEL: test_tag: test15_sub1
// CHECK: operand #0
// CHECK-NEXT: - mods: <initial>
// CHECK-NEXT: - pMods: <none>
// CHECK: operand #1
// CHECK-NEXT: - mods: test15_store1
// CHECK-NEXT: - pMods: <none>
func.func @test15(%val: !sycl_id_1, %arg0 : memref<?x!sycl_accessor_1_f32_rw_dev, 4>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_1>
  %cast = memref.cast %alloca : memref<1x!sycl_id_1> to memref<?x!sycl_id_1>  
  affine.store %val, %alloca[0] {tag_name = "test15_store1"}: memref<1x!sycl_id_1>
  %1 = sycl.accessor.subscript %arg0[%cast] {tag = "test15_sub1", ArgumentTypes = [memref<?x!sycl_accessor_1_f32_rw_dev, 4>, memref<?x!sycl_id_1>], FunctionName = @"operator[]", MangledFunctionName = @subscript, TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_rw_dev, 4>, memref<?x!sycl_id_1>) -> memref<?xf32, 4>
  return
}

// -----

// COM: The following tests perform similar checks for !llvm.ptr instead of
// memref. The numbers match the memref-based test with the corresponding
// scenario.

// CHECK-LABEL: test_tag: ptr1_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: ptr1_store1
// CHECK-NEXT:    - pMods: <none>
// CHECK-LABEL: test_tag: ptr1_load2:
// CHECK:        operand #0
// CHECK-NEXT:   - mods: ptr1_store2
// CHECK-NEXT:   - pMods: <none>
llvm.func @ptr_test1(%val : i32) {
  %cst1_i64 = arith.constant 1 : i64
  %ptr1 = llvm.alloca %cst1_i64 x i32 : (i64) -> !llvm.ptr
  %ptr2 = llvm.alloca %cst1_i64 x i32 : (i64) -> !llvm.ptr
  llvm.store %val, %ptr1 {tag_name = "ptr1_store1"} : i32, !llvm.ptr
  %1 = llvm.load %ptr1 {tag = "ptr1_load1"} : !llvm.ptr -> i32
  llvm.store %val, %ptr2 {tag_name = "ptr1_store2"} : i32, !llvm.ptr
  %2 = llvm.load %ptr2 {tag = "ptr1_load2"} : !llvm.ptr -> i32
  llvm.return
}

// CHECK-LABEL: test_tag: ptr3_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: ptr3_store1
// CHECK-NEXT:    - pMods: ptr3_store2
llvm.func @ptr_test3(%val : i32) {
  %cst1_i64 = arith.constant 1 : i64
  %ptr = llvm.alloca %cst1_i64 x i32 : (i64) -> !llvm.ptr
  %cast = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<4>
  llvm.store %val, %cast {tag_name = "ptr3_store1"} : i32, !llvm.ptr<4>
  // The following store may alias with %cast and must therefore appear in the
  // potential modifiers. This is different from the memref test above, where
  // they must alias.
  llvm.store %val, %ptr {tag_name = "ptr3_store2"} : i32, !llvm.ptr
  %2 = llvm.load %cast {tag = "ptr3_load1"} : !llvm.ptr<4> -> i32
  llvm.return
}


// CHECK-LABEL: test_tag: ptr4_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: <initial> ptr4_store2
// CHECK-NEXT:    - pMods: ptr4_store1
llvm.func @ptr_test4(%cond : i1, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr, %val : i32) {
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  llvm.store %val, %arg2 {tag_name = "ptr4_store1"} : i32, !llvm.ptr
  llvm.br ^bb3
^bb2:
  llvm.store %val, %arg1 {tag_name = "ptr4_store2"} : i32, !llvm.ptr
  llvm.br ^bb3
^bb3:
  %1 = llvm.load %arg1 {tag = "ptr4_load1"} : !llvm.ptr -> i32
  llvm.return
}


// CHECK-LABEL: test_tag: ptr5_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods:
// CHECK-DAG:             ptr5_store1
// CHECK-DAG:             ptr5_store3
// CHECK-NEXT:    - pMods:
// CHECK-DAG:             ptr5_store2
// CHECK-DAG:             ptr5_store4
llvm.func @ptr_test5(%cond : i1, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr, %val : i32) {
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  llvm.store %val, %arg1 {tag_name = "ptr5_store1"} : i32, !llvm.ptr
  llvm.store %val, %arg2 {tag_name = "ptr5_store2"} : i32, !llvm.ptr
  llvm.br ^bb3
^bb2:
  llvm.store %val, %arg1 {tag_name = "ptr5_store3"} : i32, !llvm.ptr
  llvm.store %val, %arg2 {tag_name = "ptr5_store4"} : i32, !llvm.ptr
  llvm.br ^bb3
^bb3:
  %1 = llvm.load %arg1 {tag = "ptr5_load1"} : !llvm.ptr -> i32
  llvm.return
}

// CHECK-LABEL: test_tag: ptr6_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: ptr6_store2
// CHECK-NEXT:    - pMods: <none>
llvm.func @ptr_test6(%arg1 : !llvm.ptr, %arg2 : !llvm.ptr, %val : i32) {
  llvm.store %val, %arg2 {tag_name = "ptr6_store1"} : i32, !llvm.ptr
  llvm.store %val, %arg1 {tag_name = "ptr6_store2"} : i32, !llvm.ptr
  %1 = llvm.load %arg1 {tag = "ptr6_load1"} : !llvm.ptr -> i32
  llvm.return
}


// CHECK-LABEL: test_tag: ptr9_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: <initial>
// CHECK-NEXT:    - pMods: <none>
llvm.func @ptr_test9(%arg1 : !llvm.ptr) {
  %1 = llvm.load %arg1 {tag = "ptr9_load1"} : !llvm.ptr -> i32
  llvm.return
}

llvm.func external @ptr_foo(%arg0 : !llvm.ptr) -> ()

// CHECK-LABEL: test_tag: ptr10_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: ptr10_store1
// CHECK-NEXT:    - pMods: <none>
// CHECK-LABEL: test_tag: ptr10_load2:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: <unknown>
// CHECK-NEXT:    - pMods: <unknown>
llvm.func @ptr_test10(%val : i32) {
  %cst1_i64 = arith.constant 1 : i64
  %ptr1 = llvm.alloca %cst1_i64 x i32 : (i64) -> !llvm.ptr
  llvm.store %val, %ptr1 {tag_name = "ptr10_store1"} : i32, !llvm.ptr
  %1 = llvm.load %ptr1 {tag = "ptr10_load1"} : !llvm.ptr -> i32
  llvm.call @ptr_foo(%ptr1) : (!llvm.ptr) -> ()
  %2 = llvm.load %ptr1 {tag = "ptr10_load2"} : !llvm.ptr -> i32
  llvm.return
}

// CHECK-LABEL: test_tag: ptr12_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: <initial> ptr12_store1
// CHECK-NEXT:    - pMods: <none>
// CHECK-LABEL: test_tag: ptr12_load2:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: <initial>
// CHECK-NEXT:    - pMods: ptr12_store1
llvm.func @ptr_test12(%cond : i1, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr, %val : i32) {
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  llvm.store %val, %arg1 {tag_name = "ptr12_store1"} : i32, !llvm.ptr
  llvm.br ^bb2
^bb2:
  %1 = llvm.load %arg1 {tag = "ptr12_load1"} : !llvm.ptr -> i32
  %2 = llvm.load %arg2 {tag = "ptr12_load2"} : !llvm.ptr -> i32
  llvm.return
}

// CHECK-LABEL: test_tag: ptr13_load1:
// CHECK:        operand #0
// CHECK-NEXT:    - mods: 
// CHECK-DAG:             ptr13_store1
// CHECK-DAG:             ptr13_store2
// CHECK-NEXT:    - pMods: <none>
llvm.func @ptr_test13(%cond : i1, %arg1 : !llvm.ptr, %arg2 : !llvm.ptr, %val : i32) {
  llvm.store %val, %arg1 {tag_name = "ptr13_store1"} : i32, !llvm.ptr
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  llvm.store %val, %arg1 {tag_name = "ptr13_store2"} : i32, !llvm.ptr
  llvm.br ^bb2
^bb2:
  %1 = llvm.load %arg1 {tag = "ptr13_load1"} : !llvm.ptr -> i32
  llvm.return
}

// -----

// COM: Test that reaching definition is propagated inter-procedurally.
// CHECK-LABEL: test_tag: callee_ptr
// CHECK: operand #0
// CHECK-NEXT: - mods: caller_store
// CHECK-NEXT: - pMods: <none>
func.func private @callee(%ptr: memref<i32>) -> memref<i32> {
  return {tag = "callee_ptr"} %ptr : memref<i32>
}

func.func @caller() {
  %ptr = memref.alloc() : memref<i32>
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag_name = "caller_store"} : memref<i32>
  %0 = func.call @callee(%ptr) : (memref<i32>) -> memref<i32>
  return
}

// -----

// COM: Test that the analysis does not break with non-functions

// CHECK-LABEL: test_tag: last_insert:
// CHECK-NEXT:   operand #0
// CHECK-NEXT:   - mods: <unknown>
// CHECK-NEXT:   - pMods: <unknown>
// CHECK-NEXT:   operand #1
// CHECK-NEXT:   - mods: <unknown>
// CHECK-NEXT:   - pMods: <unknown>
llvm.mlir.global internal @init_glboal() : !llvm.struct<"foo", (i8)> {
  %0 = llvm.mlir.constant(0 : i8) : i8
  %1 = llvm.mlir.undef {tag_name = "undef"} : !llvm.struct<"foo", (i8)>
  %2 = llvm.insertvalue %0, %1[0] {tag = "last_insert"} : !llvm.struct<"foo", (i8)>
  llvm.return %2 : !llvm.struct<"foo", (i8)>
}
