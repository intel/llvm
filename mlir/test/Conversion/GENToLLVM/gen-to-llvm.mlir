// RUN: mlir-opt -convert-gen-to-llvm -split-input-file %s | FileCheck %s

llvm.func @gen_special_regs() -> i32 {
  // CHECK-LABEL: gen_special_regs
  // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[CI:%.*]] = llvm.call @_Z12get_local_idj([[ZERO]]) : (i32) -> i64
  // CHECK-NEXT: llvm.trunc [[CI]] : i64 to i32
  %1 = gen.workitem.id.x : i32
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @_Z12get_local_idj([[ONE]]) : (i32) -> i64
  %2 = gen.workitem.id.y : i32
  // CHECK: [[TWO:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.call @_Z12get_local_idj([[TWO]]) : (i32) -> i64
  %3 = gen.workitem.id.z : i64

  // CHECK: [[ZERO1:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.call @_Z12get_group_idj([[ZERO1]]) : (i32) -> i64
  %4 = gen.workgroup.id.x : i32
  // CHECK: [[ONE1:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @_Z12get_group_idj([[ONE1]]) : (i32) -> i64
  %5 = gen.workgroup.id.y : i64
  // CHECK: [[TWO1:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.call @_Z12get_group_idj([[TWO1]]) : (i32) -> i64
  %6 = gen.workgroup.id.z : i32

  // CHECK: [[ZERO2:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.call @_Z14get_local_sizej([[ZERO2]]) : (i32) -> i64
  %7 = gen.workgroup.dim.x : i32
  // CHECK: [[ONE2:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @_Z14get_local_sizej([[ONE2]]) : (i32) -> i64
  %8 = gen.workgroup.dim.y : i64
  // CHECK: [[TWO2:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.call @_Z14get_local_sizej([[TWO2]]) : (i32) -> i64
  %9 = gen.workgroup.dim.z : i32

  // CHECK: [[ZERO3:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.call @_Z14get_num_groupsj([[ZERO3]]) : (i32) -> i64
  %10 = gen.grid.dim.x : i32
  // CHECK: [[ONE3:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @_Z14get_num_groupsj([[ONE3]]) : (i32) -> i64
  %11 = gen.grid.dim.y : i64
  // CHECK: [[TWO3:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.call @_Z14get_num_groupsj([[TWO3]]) : (i32) -> i64
  %12 = gen.grid.dim.z : i32
  llvm.return %1 : i32
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
  %1 = gen.sub_group_shuffle xor %0, %0 : (i32) -> i32
  %2 = gen.sub_group_shuffle up %0, %0 : (i32) -> i32
  %3 = gen.sub_group_shuffle down %0, %0 : (i32) -> i32
  %4 = gen.sub_group_shuffle idx %0, %0 : (i32) -> i32

  // CHECK: [[ZERO1:%.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorcj([[ZERO1]], [[ZERO]]) {passthrough = ["convergent"]} : (i8, i32) -> i8
  %5 = llvm.mlir.constant(0 : i8) : i8
  %6 = gen.sub_group_shuffle xor %5, %0 : (i8) -> i8

  // CHECK: [[ZERO2:%.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorsj([[ZERO2]], [[ZERO]]) {passthrough = ["convergent"]} : (i16, i32) -> i16
  %7 = llvm.mlir.constant(0 : i16) : i16
  %8 = gen.sub_group_shuffle xor %7, %0 : (i16) -> i16

  // CHECK: [[ZERO3:%.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorlj([[ZERO3]], [[ZERO]]) {passthrough = ["convergent"]} : (i64, i32) -> i64
  %9 = llvm.mlir.constant(0 : i64) : i64
  %10 = gen.sub_group_shuffle xor %9, %0 : (i64) -> i64

  // CHECK: [[ZERO4:%.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorDhj([[ZERO4]], [[ZERO]]) {passthrough = ["convergent"]} : (f16, i32) -> f16
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  %12 = gen.sub_group_shuffle xor %11, %0 : (f16) -> f16

  // CHECK: [[ZERO5:%.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK: llvm.call @_Z21sub_group_shuffle_xorfj([[ZERO5]], [[ZERO]]) {passthrough = ["convergent"]} : (f32, i32) -> f32
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  %14 = gen.sub_group_shuffle xor %13, %0 : (f32) -> f32

  // CHECK: [[ZERO6:%.*]] = llvm.mlir.constant(0.000000e+00 : f64) : f64
  // CHECK: llvm.call @_Z21sub_group_shuffle_xordj([[ZERO6]], [[ZERO]]) {passthrough = ["convergent"]} : (f64, i32) -> f64
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  %16 = gen.sub_group_shuffle xor %15, %0 : (f64) -> f64
  llvm.return
}
