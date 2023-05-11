// RUN: polygeist-opt %s --convert-polygeist-to-llvm='use-opaque-pointers=1' --split-input-file | FileCheck %s

// CHECK: llvm.func @test_struct(%arg0: !llvm.struct<(ptr, i32)>)
func.func @test_struct(%arg0: !polygeist.struct<(memref<i32>, i32)>) {
  return
}

// -----

// CHECK: llvm.func @test_struct(%arg0: !llvm.struct<"name", (ptr, i32)>)
func.func @test_struct(%arg0: !polygeist.struct<"name", (memref<i32>, i32)>) {
  return
}
