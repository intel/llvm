// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

// Ensure the functions below are not canonicalized. 

// -----

// CHECK:  func.func @SubIndex2([[A0:%.*]]: memref<?x!llvm.struct<(i32)>>, [[A1:%.*]]: index, [[A2:%.*]]: index) -> memref<i32> {
// CHECK-NEXT:     [[T0:%.*]] = "polygeist.subindex"([[A0]], [[A1]]) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
// CHECK-NEXT:     [[T1:%.*]] = "polygeist.subindex"([[T0]], [[A2]]) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:     return [[T1]] : memref<i32>
// CHECK-NEXT:  }
func.func @SubIndex2(%arg0: memref<?x!llvm.struct<(i32)>>, %arg1: index, %arg2: index) -> memref<i32> {
  %0 = "polygeist.subindex"(%arg0, %arg1) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
  %1 = "polygeist.subindex"(%0, %arg2) : (memref<?xi32>, index) -> memref<i32>
  return %1 : memref<i32>
}

// -----

// CHECK:  func.func @SubToCast([[A0:%.*]]: memref<?x!llvm.struct<(i32)>>) -> memref<?xi32> {
// CHECK-NEXT:     [[C0:%.*]] = arith.constant 0 : index
// CHECK-NEXT:     [[T0:%.*]] = "polygeist.subindex"([[A0]], [[C0]]) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
// CHECK-NEXT:     return [[T0]] : memref<?xi32>
// CHECK-NEXT:  }
func.func @SubToCast(%arg0: memref<?x!llvm.struct<(i32)>>) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK:  func.func @SimplifySubIndexUsers([[A0:%.*]]: memref<?x!llvm.struct<(i32)>>) -> memref<?xi32> {
// CHECK-NEXT:     [[C0:%.*]] = arith.constant 0 : index
// CHECK-NEXT:     [[C1:%.*]] = arith.constant 0 : i32
// CHECK-NEXT:     [[T0:%.*]] = "polygeist.subindex"([[A0]], [[C0]]) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
// CHECK-NEXT:     memref.store [[C1]], [[T0]][[[C0]]] : memref<?xi32>
// CHECK-NEXT:     return [[T0]] : memref<?xi32>
// CHECK-NEXT:  }
func.func @SimplifySubIndexUsers(%arg0: memref<?x!llvm.struct<(i32)>>) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
  memref.store %c0_i32, %0[%c0] : memref<?xi32>
  return %0 : memref<?xi32>
}

// -----
