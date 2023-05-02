// RUN: polygeist-opt -split-input-file -test-reaching-definition %s 2>&1 | FileCheck %s

// COM: Test that the correct reaching definition reaches 2 loads.
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

// COM: Test that loads has the correct reaching definitions in the presence of a cast.
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

// COM: Test that a (must) aliased definition kills a previous aliased definition.
// CHECK-LABEL: test_tag: test3_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test3_store2
// CHECK-NEXT: - pMods:
func.func @test3(%val : i32, %idx: index) {
  %alloca = memref.alloca()  : memref<1xi32>
  %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
  %addrspace_cast = memref.memory_space_cast %cast : memref<?xi32> to memref<?xi32, 4>  
  memref.store %val, %addrspace_cast[%idx] {tag_name = "test3_store1"} : memref<?xi32, 4>
  memref.store %val, %alloca[%idx] {tag_name = "test3_store2"} : memref<1xi32>  
  %1 = memref.load %cast[%idx] {tag = "test3_load1"} : memref<?xi32>
  return
}

// COM: Test that "must" and "may" aliased definitions reach a load.
// CHECK-LABEL: test_tag: test4_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test4_store1 test4_store3
// CHECK-NEXT: - pMods: test4_store2 test4_store4
func.func @test4(%cond: i1, %arg1: memref<i32>, %arg2: memref<i32>) {
  scf.if %cond {
    %c0 = arith.constant 0 : i32
    memref.store %c0, %arg1[] {tag_name = "test4_store1"}: memref<i32>
    memref.store %c0, %arg2[] {tag_name = "test4_store2"}: memref<i32>
  } else {
    %c1 = arith.constant 1 : i32
    memref.store %c1, %arg1[] {tag_name = "test4_store3"}: memref<i32>    
    memref.store %c1, %arg2[] {tag_name = "test4_store4"} : memref<i32>
  }
  %1 = memref.load %arg1[] {tag = "test4_load1"} : memref<i32>
  return
}

// CHECK-LABEL: test_tag: test5_load1
// CHECK: operand #0
// CHECK-NEXT: - mods: test5_store2 test5_store3
// CHECK-NEXT: - pMods: test5_store4
func.func @test5(%cond: i1, %arg1: memref<i32>, %arg2: memref<i32>) {
  scf.if %cond {
    %c0 = arith.constant 0 : i32
    memref.store %c0, %arg2[] {tag_name = "test5_store1"}: memref<i32>
    memref.store %c0, %arg1[] {tag_name = "test5_store2"}: memref<i32> 
  } else {
    %c1 = arith.constant 1 : i32
    memref.store %c1, %arg1[] {tag_name = "test5_store3"} : memref<i32>    
    memref.store %c1, %arg2[] {tag_name = "test5_store4"}: memref<i32>    
  }
  %1 = memref.load %arg1[] {tag = "test5_load1"} : memref<i32>
  return
}
