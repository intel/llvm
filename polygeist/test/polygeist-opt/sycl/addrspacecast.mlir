// RUN: polygeist-opt --convert-polygeist-to-llvm="use-bare-ptr-memref-call-conv" --split-input-file %s | FileCheck %s

// CHECK-LABEL: llvm.func @test_1(%arg0: !llvm.ptr<i32>) -> !llvm.ptr<i32, 4> {
// CHECK-NEXT:    %0 = llvm.addrspacecast %arg0 : !llvm.ptr<i32> to !llvm.ptr<i32, 4>
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr<i32, 4>
// CHECK-NEXT:  }

func.func @test_1(%arg0: memref<?xi32>) -> memref<?xi32, 4> {
  %0 = "polygeist.addrspacecast"(%arg0) : (memref<?xi32>) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}
