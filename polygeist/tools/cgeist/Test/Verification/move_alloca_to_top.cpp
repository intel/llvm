// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: define void @move_alloca(float* %0) {
// CHECK-DAG:     %{{.*}} = alloca float, i8 1, align 4
// CHECK-DAG:     %{{.*}} = alloca float, i8 1, align 4
// CHECK-NEXT:    %4 = load float, float* %0, align 4
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }

llvm.func @move_alloca(%arg0: !llvm.ptr<f32>) {
    %0 = llvm.load %arg0 : !llvm.ptr<f32>
    %1 = llvm.mlir.constant(1 : i8) : i8
    %2 = llvm.alloca %1 x f32 : (i8) -> !llvm.ptr<f32>
    %3 = llvm.alloca %1 x f32 : (i8) -> !llvm.ptr<f32>
    llvm.return
}
