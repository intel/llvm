// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module {
  func.func @ll(%arg0: !llvm.ptr) -> !llvm.ptr {
    %c1_i64 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %arg0, %2 : !llvm.ptr, !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.ptr
	return %3 : !llvm.ptr
  }
}

// CHECK:   func.func @ll(%arg0: !llvm.ptr) -> !llvm.ptr {
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     return %arg0 : !llvm.ptr
// CHECK-NEXT:   }

// -----

// COM: For the following example, mem2reg should not perform promotion of the
//      alloca, as the types of the store and load do not match the allocatd
//      type.
module {
  func.func @no_promotion(%arg0 : f32) -> i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<"union.anon", (i32)> {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 4 : i64} : f32, !llvm.ptr
    %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %2 : i32
  }
}

// CHECK-LABEL:   func.func @no_promotion(
// CHECK-SAME:                            %[[VAL_0:.*]]: f32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"union.anon", (i32)> {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_2]] {alignment = 4 : i64} : f32, !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK:           llvm.return %[[VAL_3]] : i32
// CHECK:         }
