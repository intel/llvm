// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

// -----

// CHECK:  func.func @main(%arg0: index) -> memref<1000xi32> {
// CHECK:    %alloca = memref.alloca() : memref<2x1000xi32>
// CHECK:    %0 = "polygeist.subindex"(%alloca, %arg0) : (memref<2x1000xi32>, index) -> memref<1000xi32>
// CHECK:    return %0 : memref<1000xi32>
// CHECK:  }
func.func @main(%arg0 : index) -> memref<1000xi32> {
  %c0 = arith.constant 0 : index
  %1 = memref.alloca() : memref<2x1000xi32>
    %3 = "polygeist.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) -> memref<?x1000xi32>
    %4 = "polygeist.subindex"(%3, %c0) : (memref<?x1000xi32>, index) -> memref<1000xi32>
  return %4 : memref<1000xi32>
}
  
// -----
  
  func.func @fold2ref(%arg0 : !llvm.ptr) -> memref<?xi32> {
        %c0_i32 = arith.constant 0 : i32
        %11 = llvm.getelementptr %arg0[%c0_i32, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, i32)>
        %12 = "polygeist.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi32>
    return %12 : memref<?xi32>
  }

// CHECK:   func.func @fold2ref(%arg0: !llvm.ptr) -> memref<?xi32> {
// CHECK-NEXT:     %0 = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:     return %0 : memref<?xi32>
// CHECK-NEXT:   }

  func.func @nofold2ref(%arg0 : !llvm.ptr) -> memref<?xi32> {
        %c0_i32 = arith.constant 0 : i32
        %11 = llvm.getelementptr %arg0[%c0_i32, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, i32)>
        %12 = "polygeist.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi32>
    return %12 : memref<?xi32>
  }

// CHECK: @nofold2ref(%arg0: !llvm.ptr) -> memref<?xi32> {
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:     return %1 : memref<?xi32>
// CHECK-NEXT:   }

func.func @memref2ptr(%arg0: memref<10xi32>) -> !llvm.ptr {
     %c2 = arith.constant 2 : index
     %0 = "polygeist.subindex"(%arg0, %c2) : (memref<10xi32>, index) -> memref<?xi32>
     %1 = "polygeist.memref2pointer"(%0) : (memref<?xi32>) -> !llvm.ptr
     return %1 : !llvm.ptr
}
// CHECK-LABEL:   func.func @memref2ptr(
// CHECK-SAME:                          %[[VAL_0:.*]]: memref<10xi32>) -> !llvm.ptr {
// CHECK-NEXT:      %[[VAL_1:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<10xi32>) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_1]][2] : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      return %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:    }
