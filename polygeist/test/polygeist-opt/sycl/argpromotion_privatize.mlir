// RUN: polygeist-opt --arg-promotion %s | FileCheck %s

gpu.module @device_func {

  // COM: Test that both @callee and @wrapper are privatized.
  // CHECK-DAG: func.func private @callee(%arg0: memref<?xi32> {llvm.noalias}, %arg1: memref<?xi64> {llvm.noalias}) -> i64 attributes {llvm.linkage = #llvm.linkage<private>}
  // CHECK-DAG: func.func private @wrapper(%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<private>}

  func.func @callee(%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi32>
    %1 = "polygeist.subindex"(%arg0, %c1) : (memref<?x!llvm.struct<(i32, i64)>>, index) -> memref<?xi64>
    %2 = affine.load %0[0] : memref<?xi32>
    %3 = affine.load %1[0] : memref<?xi64>
    %4 = arith.extsi %2 : i32 to i64
    %5 = arith.addi %3, %4 : i64
    return %5 : i64
  }

  gpu.func @caller1() kernel {
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast = memref.cast %alloca : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    func.call @callee(%cast) : (memref<?x!llvm.struct<(i32, i64)>>) -> i64
    gpu.return
  }

  gpu.func @caller2() kernel {
    %alloca = memref.alloca() : memref<3x!llvm.struct<(i32, i64)>>
    %cast = memref.cast %alloca : memref<3x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    func.call @wrapper(%cast) : (memref<?x!llvm.struct<(i32, i64)>>) -> (i64)
    gpu.return
  }

  func.func @wrapper(%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = func.call @callee(%arg0) : (memref<?x!llvm.struct<(i32, i64)>>) -> i64
    return %0 : i64
  }

}
