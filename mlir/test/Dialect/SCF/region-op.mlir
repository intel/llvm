// RUN: mlir-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

func.func @test() {
  scf.env_region "test" {
    scf.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   scf.env_region "test" {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test() {
  scf.env_region "test" {
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-NEXT:   scf.env_region "test" {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index) {
  scf.env_region "test" %arg1 : index {
    scf.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   scf.env_region "test" %[[ARG1]] : index {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index, %arg2: i64) {
  scf.env_region "test" %arg1, %arg2 : index, i64 {
    scf.env_region_yield
  }
  return
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT:   scf.env_region "test" %[[ARG1]], %[[ARG2]] : index, i64 {
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @test(%arg1: index) -> index {
  %0 = scf.env_region "test" -> index {
    scf.env_region_yield %arg1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = scf.env_region "test" -> index {
//  CHECK-NEXT:     scf.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index

// -----

func.func @test(%arg1: index) -> index {
  %0 = scf.env_region "test" %arg1 : index -> index {
    scf.env_region_yield %arg1: index
  }
  return %0: index
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index)
//  CHECK-NEXT:   %[[RES:.*]] = scf.env_region "test" %[[ARG1]] : index -> index {
//  CHECK-NEXT:     scf.env_region_yield %[[ARG1]] : index
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %[[RES]] : index
