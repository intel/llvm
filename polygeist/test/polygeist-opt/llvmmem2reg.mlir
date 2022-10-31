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

// -----

module {
  func.func @mixed(%mr : !llvm.ptr<memref<2xf32>>) {
    %2 = memref.alloc() : memref<2xf32>
    llvm.store %2, %mr : !llvm.ptr<memref<2xf32>>
    return
  }
}

// CHECK:   func.func @mixed(%arg0: !llvm.ptr<memref<2xf32>>)
// CHECK-NEXT:     %alloc = memref.alloc() : memref<2xf32>
// CHECK-NEXT:     llvm.store %alloc, %arg0 : !llvm.ptr<memref<2xf32>>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
