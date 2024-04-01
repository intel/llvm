// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

llvm.func @gen_nd_range(%dim: i32) {
  // CHECK-LABEL: gen_nd_range
  // CHECK-SAME:              (%[[DIM:.*]]: i32)
  // CHECK: gen.local_id %[[DIM]]
  %0 = gen.local_id %dim
  // CHECK: gen.work_group_id %[[DIM]]
  %1 = gen.work_group_id %dim
  // CHECK: gen.work_group_size %[[DIM]]
  %2 = gen.work_group_size %dim
  // CHECK: gen.num_work_groups %[[DIM]]
  %3 = gen.num_work_groups %dim
  llvm.return
}

llvm.func @gen.barrier() {
  // CHECK-LABEL: gen.barrier
  // CHECK: gen.barrier
  gen.barrier
  llvm.return
}

llvm.func @gen.sub_group_shuffle() {
  // CHECK-LABEL: gen.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = gen.sub_group_shuffle xor %0, %0 : i32
  %1 = gen.sub_group_shuffle xor %0, %0 : i32
  // CHECK: %2 = gen.sub_group_shuffle up %0, %0 : i32
  %2 = gen.sub_group_shuffle up %0, %0 : i32
  // CHECK: %3 = gen.sub_group_shuffle down %0, %0 : i32
  %3 = gen.sub_group_shuffle down %0, %0 : i32
  // CHECK: %4 = gen.sub_group_shuffle idx %0, %0 : i32
  %4 = gen.sub_group_shuffle idx %0, %0 : i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %6 = gen.sub_group_shuffle xor %5, %0 : i8
  %6 = gen.sub_group_shuffle xor %5, %0 : i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %8 = gen.sub_group_shuffle xor %7, %0 : i16
  %8 = gen.sub_group_shuffle xor %7, %0 : i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %10 = gen.sub_group_shuffle xor %9, %0 : i64
  %10 = gen.sub_group_shuffle xor %9, %0 : i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %12 = gen.sub_group_shuffle xor %11, %0 : f16
  %12 = gen.sub_group_shuffle xor %11, %0 : f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %14 = gen.sub_group_shuffle xor %13, %0 : f32
  %14 = gen.sub_group_shuffle xor %13, %0 : f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %16 = gen.sub_group_shuffle xor %15, %0 : f64
  %16 = gen.sub_group_shuffle xor %15, %0 : f64
  llvm.return
}
