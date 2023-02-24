// RUN: polygeist-opt --convert-polygeist-to-llvm="use-bare-ptr-memref-call-conv" --split-input-file %s | FileCheck %s

// CHECK-LABEL: llvm.func @test_1(%arg0: !llvm.ptr<i32>) -> !llvm.ptr<i32, 4> {
// CHECK-NEXT:    %0 = llvm.bitcast %arg0 : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK-NEXT:    %1 = llvm.addrspacecast %0 : !llvm.ptr<i32> to !llvm.ptr<i32, 4>
// CHECK-NEXT:    %2 = llvm.bitcast %1 : !llvm.ptr<i32, 4> to !llvm.ptr<i32, 4>
// CHECK-NEXT:    llvm.return %2 : !llvm.ptr<i32, 4>
// CHECK-NEXT:  }

func.func @test_1(%arg0: memref<?xi32>) -> memref<?xi32, 4> {
  %0 = "polygeist.addrspacecast"(%arg0) : (memref<?xi32>) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}
