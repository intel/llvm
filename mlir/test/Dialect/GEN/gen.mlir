// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

llvm.func @gen_special_regs() -> i32 {
  // CHECK-LABEL: gen_special_regs
  // CHECK: gen.workitem.id.x : i32
  %0 = gen.workitem.id.x : i32
  // CHECK: gen.workitem.id.y : i32
  %1 = gen.workitem.id.y : i32
  // CHECK: gen.workitem.id.z : i32
  %2 = gen.workitem.id.z : i32
  // CHECK: gen.workgroup.id.x : i32
  %3 = gen.workgroup.id.x : i32
  // CHECK: gen.workgroup.id.y : i32
  %4 = gen.workgroup.id.y : i32
  // CHECK: gen.workgroup.id.z : i32
  %5 = gen.workgroup.id.z : i32
  // CHECK: gen.workgroup.dim.x : i32
  %6 = gen.workgroup.dim.x : i32
  // CHECK: gen.workgroup.dim.y : i32
  %7 = gen.workgroup.dim.y : i32
  // CHECK: gen.workgroup.dim.z : i32
  %8 = gen.workgroup.dim.z : i32
  // CHECK: gen.grid.dim.x : i32
  %9 = gen.grid.dim.x : i32
  // CHECK: gen.grid.dim.y : i32
  %10 = gen.grid.dim.y : i32
  // CHECK: gen.grid.dim.z : i32
  %11 = gen.grid.dim.z : i32
  llvm.return %0 : i32
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
  // CHECK: %1 = gen.sub_group_shuffle xor %0, %0 : i32 -> i32
  %1 = gen.sub_group_shuffle xor %0, %0 : i32 -> i32
  // CHECK: %2 = gen.sub_group_shuffle up %0, %0 : i32 -> i32
  %2 = gen.sub_group_shuffle up %0, %0 : i32 -> i32
  // CHECK: %3 = gen.sub_group_shuffle down %0, %0 : i32 -> i32
  %3 = gen.sub_group_shuffle down %0, %0 : i32 -> i32
  // CHECK: %4 = gen.sub_group_shuffle idx %0, %0 : i32 -> i32
  %4 = gen.sub_group_shuffle idx %0, %0 : i32 -> i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %6 = gen.sub_group_shuffle xor %5, %0 : i8 -> i8
  %6 = gen.sub_group_shuffle xor %5, %0 : i8 -> i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %8 = gen.sub_group_shuffle xor %7, %0 : i16 -> i16
  %8 = gen.sub_group_shuffle xor %7, %0 : i16 -> i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %10 = gen.sub_group_shuffle xor %9, %0 : i64 -> i64
  %10 = gen.sub_group_shuffle xor %9, %0 : i64 -> i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %12 = gen.sub_group_shuffle xor %11, %0 : f16 -> f16
  %12 = gen.sub_group_shuffle xor %11, %0 : f16 -> f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %14 = gen.sub_group_shuffle xor %13, %0 : f32 -> f32
  %14 = gen.sub_group_shuffle xor %13, %0 : f32 -> f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %16 = gen.sub_group_shuffle xor %15, %0 : f64 -> f64
  %16 = gen.sub_group_shuffle xor %15, %0 : f64 -> f64
  llvm.return
}
