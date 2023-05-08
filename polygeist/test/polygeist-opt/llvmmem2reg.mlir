// RUN: polygeist-opt --mem2reg --split-input-file %s | FileCheck %s

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
