// RUN: polygeist-opt -split-input-file -test-reaching-definition %s 2>&1 | FileCheck %s

// COM: Test that the correct definition reaches a load.
// CHECK-LABEL: test_tag: test1_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test1_store1
// CHECK-NEXT: - pMods: 
// CHECK-LABEL: test_tag: test1_load2
// CHECK: operand #0
// CHECK-NEXT: - mods: test1_store2
// CHECK-NEXT: - pMods: 
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
// CHECK-NEXT: - pMods: 
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
// CHECK-NEXT: - pMods:
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
// CHECK-NEXT: - mods: test4_store2
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
// CHECK-NEXT: - mods: test5_store1 test5_store3
// CHECK-NEXT: - pMods: test5_store2 test5_store4
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
// CHECK-NEXT: - pMods:
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
// CHECK-NEXT: - mods:
// CHECK-NEXT: - pMods: test7_store1
// CHECK: operand #0
// CHECK-NEXT: - mods:
// CHECK-NEXT: - pMods:
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
// CHECK-NEXT: - pMods:
// CHECK-LABEL: test_tag: test8_load2
// CHECK: operand #0
// CHECK-NEXT: - mods:
// CHECK-NEXT: - pMods:
// CHECK-LABEL: test_tag: test8_load3
// CHECK: operand #0
// CHECK-NEXT: - mods:
// CHECK-NEXT: - pMods:
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

// COM: Ensure definitions and potential definitions are unknown if operation 
//       has stored or allocated a value used by a load.
// CHECK-LABEL: test_tag: test9_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: <unknown>
// CHECK-NEXT: - pMods: <unknown>
func.func @test9(%arg1: memref<i32>) {
  %1 = memref.load %arg1[] {tag = "test9_load1"} : memref<i32>
  return
}

// COM: Test that a operation with unknown side effects kills a previous definition.
// CHECK-LABEL: test_tag: test10_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test10_store1
// CHECK-NEXT: - pMods:
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
// CHECK-NEXT: - mods: test11_store2 test11_store1
// CHECK-NEXT: - pMods:
// CHECK-LABEL: test_tag: test11_load2
// CHECK: operand #0
// CHECK-NEXT: - mods:
// CHECK-NEXT: - pMods:
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
