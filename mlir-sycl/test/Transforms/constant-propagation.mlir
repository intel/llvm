// RUN: sycl-mlir-opt -pass-pipeline='builtin.module(gpu.module(gpu.func(sycl-constant-propagation)))' %s -mlir-pass-statistics 2>&1 | FileCheck %s

// COM: Check we can detect %c is constant and it's dropped.

// CHECK:      'gpu.module' Pipeline
// CHECK-NEXT:   'gpu.func' Pipeline
// CHECK-NEXT:     ConstantPropagationPass
// CHECK-NEXT:       (S) 1 num-propagated-constants - Number of propagated constants

gpu.module @kernels {
// CHECK-LABEL:     gpu.func @k0(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<1xi64>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i64,
// CHECK-SAME:                   %[[VAL_2:.*]]: i64) kernel {
// CHECK-NEXT:        %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:        %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:        %[[VAL_5:.*]] = arith.muli %[[VAL_4]], %[[VAL_2]] : i64
// CHECK-NEXT:        affine.store %[[VAL_5]], %[[VAL_0]][0] : memref<1xi64>
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k0(%res: memref<1xi64>, %x: i64, %c: i64, %y: i64) kernel {
    %add = arith.addi %x, %c : i64
    %mul = arith.muli %add, %y : i64
    affine.store %mul, %res[0] : memref<1xi64>
    gpu.return
  }
}

// CHECK-LABEL:   llvm.func internal @foo(
// CHECK-SAME:                            %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                            %[[VAL_1:.*]]: i64,
// CHECK-SAME:                            %[[VAL_2:.*]]: i64) {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      sycl.host.schedule_kernel @kernels::@k0(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func internal @foo(%res: !llvm.ptr, %x: i64, %y: i64) {
  %c = llvm.mlir.constant(0 : i64) : i64
  sycl.host.schedule_kernel @kernels::@k0(%res, %x, %c, %y)
      : (!llvm.ptr, i64, i64, i64) -> ()
  llvm.return
}
