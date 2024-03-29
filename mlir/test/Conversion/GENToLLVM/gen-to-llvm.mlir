// RUN: mlir-opt -convert-gen-to-llvm -split-input-file %s | FileCheck %s

llvm.func @gen_nd_range(%dim: i32) {
  // CHECK-LABEL: gen_nd_range
  // CHECK-SAME:              (%[[DIM:.*]]: i32)
  // CHECK:         llvm.call @_Z12get_local_idj(%[[DIM]]) : (i32) -> i64
  %0 = gen.local_id %dim
  // CHECK:         llvm.call @_Z12get_group_idj(%[[DIM]]) : (i32) -> i64
  %1 = gen.work_group_id %dim
  // CHECK:         llvm.call @_Z14get_local_sizej(%[[DIM]]) : (i32) -> i64
  %2 = gen.work_group_size %dim
  // CHECK:         llvm.call @_Z14get_num_groupsj(%[[DIM]]) : (i32) -> i64
  %3 = gen.num_work_groups %dim
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {passthrough = ["convergent"]}

llvm.func @gen.barrier() {
  // CHECK-LABEL: gen.barrier
  // CHECK: [[CST:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @_Z7barrierj([[CST]]) {passthrough = ["convergent"]} : (i32) -> ()
  gen.barrier
  llvm.return
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z21sub_group_shuffle_xordj(f64, i32) -> f64 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z21sub_group_shuffle_xorfj(f32, i32) -> f32 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z21sub_group_shuffle_xorDhj(f16, i32) -> f16 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z21sub_group_shuffle_xorlj(i64, i32) -> i64 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z21sub_group_shuffle_xorsj(i16, i32) -> i16 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z21sub_group_shuffle_xorcj(i8, i32) -> i8 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z17sub_group_shuffleij(i32, i32) -> i32 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z22sub_group_shuffle_downij(i32, i32) -> i32 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z20sub_group_shuffle_upij(i32, i32) -> i32 attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z21sub_group_shuffle_xorij(i32, i32) -> i32 attributes {passthrough = ["convergent"]}

llvm.func @gen.sub_group_shuffle() {
  // CHECK-LABEL: gen.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorij([[ZERO]], [[ZERO]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.call @_Z20sub_group_shuffle_upij([[ZERO]], [[ZERO]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.call @_Z22sub_group_shuffle_downij([[ZERO]], [[ZERO]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.call @_Z17sub_group_shuffleij([[ZERO]], [[ZERO]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  %1 = gen.sub_group_shuffle xor %0, %0 : i32
  %2 = gen.sub_group_shuffle up %0, %0 : i32
  %3 = gen.sub_group_shuffle down %0, %0 : i32
  %4 = gen.sub_group_shuffle idx %0, %0 : i32

  // CHECK: [[ZERO1:%.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorcj([[ZERO1]], [[ZERO]]) {passthrough = ["convergent"]} : (i8, i32) -> i8
  %5 = llvm.mlir.constant(0 : i8) : i8
  %6 = gen.sub_group_shuffle xor %5, %0 : i8

  // CHECK: [[ZERO2:%.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorsj([[ZERO2]], [[ZERO]]) {passthrough = ["convergent"]} : (i16, i32) -> i16
  %7 = llvm.mlir.constant(0 : i16) : i16
  %8 = gen.sub_group_shuffle xor %7, %0 : i16

  // CHECK: [[ZERO3:%.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorlj([[ZERO3]], [[ZERO]]) {passthrough = ["convergent"]} : (i64, i32) -> i64
  %9 = llvm.mlir.constant(0 : i64) : i64
  %10 = gen.sub_group_shuffle xor %9, %0 : i64

  // CHECK: [[ZERO4:%.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorDhj([[ZERO4]], [[ZERO]]) {passthrough = ["convergent"]} : (f16, i32) -> f16
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  %12 = gen.sub_group_shuffle xor %11, %0 : f16

  // CHECK: [[ZERO5:%.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorfj([[ZERO5]], [[ZERO]]) {passthrough = ["convergent"]} : (f32, i32) -> f32
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  %14 = gen.sub_group_shuffle xor %13, %0 : f32

  // CHECK: [[ZERO6:%.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
  // CHECK: llvm.call @_Z21sub_group_shuffle_xordj([[ZERO6]], [[ZERO]]) {passthrough = ["convergent"]} : (f64, i32) -> f64
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  %16 = gen.sub_group_shuffle xor %15, %0 : f64
  llvm.return
}
