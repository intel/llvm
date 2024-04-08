// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-gen-to-llvm))" -split-input-file %s \
// RUN: | FileCheck --check-prefixes=CHECK-64,CHECK %s
// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-gen-to-llvm{index-bitwidth=32}))" -split-input-file %s \
// RUN: | FileCheck --check-prefixes=CHECK-32,CHECK %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass.
// RUN: mlir-opt --convert-to-llvm="filter-dialects=gen" --split-input-file %s | FileCheck %s

func.func @gen_nd_range(%dim: i32) {
  // CHECK-LABEL: gen_nd_range
  // CHECK-SAME:              ([[DIM:%.*]]: i32)
  // CHECK-64:      llvm.call @_Z12get_local_idj([[DIM]]) : (i32) -> i64
  // CHECK-32:      llvm.call @_Z12get_local_idj([[DIM]]) : (i32) -> i32
  %0 = gen.local_id %dim
  // CHECK-64:      llvm.call @_Z12get_group_idj([[DIM]]) : (i32) -> i64
  // CHECK-32:      llvm.call @_Z12get_group_idj([[DIM]]) : (i32) -> i32
  %1 = gen.work_group_id %dim
  // CHECK-64:      llvm.call @_Z14get_local_sizej([[DIM]]) : (i32) -> i64
  // CHECK-32:      llvm.call @_Z14get_local_sizej([[DIM]]) : (i32) -> i32
  %2 = gen.work_group_size %dim
  // CHECK-64:      llvm.call @_Z14get_num_groupsj([[DIM]]) : (i32) -> i64
  // CHECK-32:      llvm.call @_Z14get_num_groupsj([[DIM]]) : (i32) -> i32
  %3 = gen.num_work_groups %dim
  func.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {passthrough = ["convergent"]}

func.func @gen.barrier() {
  // CHECK-LABEL: gen.barrier
  // CHECK: [[CST:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @_Z7barrierj([[CST]]) {passthrough = ["convergent"]} : (i32) -> ()
  gen.barrier
  func.return
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

func.func @gen.sub_group_shuffle() {
  // CHECK-LABEL: gen.sub_group_shuffle
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8
  %c0_i16 = arith.constant 0 : i16
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f64
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant 0.000000e+00 : f32

  // CHECK-DAG: [[ZERO:%.*]] = arith.constant 0 : i32
  // CHECK-DAG: [[ZERO1:%.*]] = arith.constant 0 : i8
  // CHECK-DAG: [[ZERO2:%.*]] = arith.constant 0 : i16
  // CHECK-DAG: [[ZERO3:%.*]] = arith.constant 0 : i64
  // CHECK-DAG: [[ZERO4:%.*]] = arith.constant 0.000000e+00 : f16
  // CHECK-DAG: [[ZERO5:%.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[ZERO6:%.*]] = arith.constant 0.000000e+00 : f64

  // CHECK: llvm.call @_Z21sub_group_shuffle_xorij([[ZERO]], [[ZERO]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.call @_Z20sub_group_shuffle_upij([[ZERO]], [[ZERO]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.call @_Z22sub_group_shuffle_downij([[ZERO]], [[ZERO]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  // CHECK: llvm.call @_Z17sub_group_shuffleij([[ZERO]], [[ZERO]]) {passthrough = ["convergent"]} : (i32, i32) -> i32
  %0 = gen.sub_group_shuffle xor %c0_i32, %c0_i32 : i32
  %1 = gen.sub_group_shuffle up %c0_i32, %c0_i32 : i32
  %2 = gen.sub_group_shuffle down %c0_i32, %c0_i32 : i32
  %3 = gen.sub_group_shuffle idx %c0_i32, %c0_i32 : i32

  // CHECK: llvm.call @_Z21sub_group_shuffle_xorcj([[ZERO1]], [[ZERO]]) {passthrough = ["convergent"]} : (i8, i32) -> i8
  %4 = gen.sub_group_shuffle xor %c0_i8, %c0_i32 : i8

  // CHECK: llvm.call @_Z21sub_group_shuffle_xorsj([[ZERO2]], [[ZERO]]) {passthrough = ["convergent"]} : (i16, i32) -> i16
  %5 = gen.sub_group_shuffle xor %c0_i16, %c0_i32 : i16

  // CHECK: llvm.call @_Z21sub_group_shuffle_xorlj([[ZERO3]], [[ZERO]]) {passthrough = ["convergent"]} : (i64, i32) -> i64
  %6 = gen.sub_group_shuffle xor %c0_i64, %c0_i32 : i64

  // CHECK: llvm.call @_Z21sub_group_shuffle_xorDhj([[ZERO4]], [[ZERO]]) {passthrough = ["convergent"]} : (f16, i32) -> f16
  %7 = gen.sub_group_shuffle xor %cst_0, %c0_i32 : f16

  // CHECK: llvm.call @_Z21sub_group_shuffle_xorfj([[ZERO5]], [[ZERO]]) {passthrough = ["convergent"]} : (f32, i32) -> f32
  %8 = gen.sub_group_shuffle xor %cst_1, %c0_i32 : f32

  // CHECK: llvm.call @_Z21sub_group_shuffle_xordj([[ZERO6]], [[ZERO]]) {passthrough = ["convergent"]} : (f64, i32) -> f64
  %9 = gen.sub_group_shuffle xor %cst, %c0_i32 : f64
  func.return
}
