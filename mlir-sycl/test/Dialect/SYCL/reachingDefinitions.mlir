// RUN: sycl-mlir-opt -split-input-file -test-reaching-definition %s 2>&1 | FileCheck %s

// CHECK-LABEL: test_tag: test1_load1
// CHECK: operand #0
// CHECK-NEXT: - a
// CHECK-LABEL: test_tag: test1_load2
// CHECK: operand #0
// CHECK-NEXT: - b
func.func @test1(%val : i32 ) {
  %alloca = memref.alloca()  : memref<1xi32>
  %alloca_0 = memref.alloca()  : memref<1xi32>
  %c0 = arith.constant 0 : index

  memref.store %val, %alloca[%c0] {tag_name = "a"} : memref<1xi32>
  %1 = memref.load %alloca[%c0] {tag = "test1_load1"} : memref<1xi32>
  memref.store %1, %alloca_0[%c0] {tag_name = "b"} : memref<1xi32>
  %2 = memref.load %alloca_0[%c0] {tag = "test1_load2"} : memref<1xi32>
  return
}

// CHECK-LABEL: test_tag: test2_load1
// CHECK: operand #0
// CHECK-NEXT: - a
// CHECK-LABEL: test_tag: test2_load2
// CHECK: operand #0
// CHECK-NEXT: - b
func.func @test2(%val : i32 ) {
  %alloca = memref.alloca()  : memref<1xi32>
  %alloca_0 = memref.alloca()  : memref<1xi32>
  %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
  %cast_0 = memref.cast %alloca_0 : memref<1xi32> to memref<?xi32>
  %c0 = arith.constant 0 : index

  memref.store %val, %cast[%c0] {tag_name = "a"} : memref<?xi32>
  %1 = memref.load %cast[%c0] {tag = "test2_load1"} : memref<?xi32>
  memref.store %1, %cast_0[%c0] {tag_name = "b"} : memref<?xi32>
  %2 = memref.load %cast_0[%c0] {tag = "test2_load2"} : memref<?xi32>
  return
}


// CHECK-LABEL: test_tag: test3_load1
// CHECK: operand #0
// CHECK-NEXT: - a
// CHECK-LABEL: test_tag: test3_load2
// CHECK: operand #0
// CHECK-NEXT: - b
func.func @test3(%val : i32 ) {
  %alloca = memref.alloca()  : memref<1xi32>
  %alloca_0 = memref.alloca()  : memref<1xi32>
  %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
  %cast_0 = memref.cast %alloca_0 : memref<1xi32> to memref<?xi32>
  %addrspace_cast = memref.memory_space_cast %cast : memref<?xi32> to memref<?xi32, 4>
  %c0 = arith.constant 0 : index

  memref.store %val, %addrspace_cast[%c0] {tag_name = "a"} : memref<?xi32, 4>
  %1 = memref.load %cast[%c0] {tag = "test3_load1"} : memref<?xi32>
  memref.store %1, %cast_0[%c0] {tag_name = "b"} : memref<?xi32>
  %2 = memref.load %cast_0[%c0] {tag = "test3_load2"} : memref<?xi32>
  return
}
