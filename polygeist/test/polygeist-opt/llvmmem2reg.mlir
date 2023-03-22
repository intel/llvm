// RUN: polygeist-opt --mem2reg --split-input-file %s | FileCheck %s

module {
  func.func @ll(%arg0: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
    %c1_i64 = arith.constant 1 : i64
    %2 = llvm.alloca %c1_i64 x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
    llvm.store %arg0, %2 : !llvm.ptr<ptr<i8>>
    %3 = llvm.load %2 : !llvm.ptr<ptr<i8>>
	return %3 : !llvm.ptr<i8>
  }
}

// CHECK:   func.func @ll(%arg0: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     return %arg0 : !llvm.ptr<i8>
// CHECK-NEXT:   }
