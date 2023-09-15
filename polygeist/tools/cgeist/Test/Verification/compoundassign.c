// RUN: cgeist %s --function=* -S -o - | FileCheck %s

#include <stdbool.h>

// CHECK-LABEL:   func.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_3:.*]] = arith.remsi %[[VAL_2]], %[[VAL_1]] : i32
// CHECK:           affine.store %[[VAL_3]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }

int f1(int *a, int b) {
  return *a %= b;
}

// CHECK-LABEL:   func.func @f2(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xf32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: i64,
// CHECK-SAME:                  %[[VAL_2:.*]]: f64) -> f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2.000000e+00 : f64
// CHECK:           %[[VAL_4:.*]] = arith.subf %[[VAL_2]], %[[VAL_3]] : f64
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK:           %[[VAL_6:.*]] = affine.load %[[VAL_0]][symbol(%[[VAL_5]])] : memref<?xf32>
// CHECK:           %[[VAL_7:.*]] = arith.extf %[[VAL_6]] : f32 to f64
// CHECK:           %[[VAL_8:.*]] = arith.subf %[[VAL_7]], %[[VAL_4]] : f64
// CHECK:           %[[VAL_9:.*]] = arith.truncf %[[VAL_8]] : f64 to f32
// CHECK:           affine.store %[[VAL_9]], %[[VAL_0]][symbol(%[[VAL_5]])] : memref<?xf32>
// CHECK:           return %[[VAL_9]] : f32
// CHECK:         }

float f2(float *a, unsigned long long index, double b) {
  return a[index] -= b - 2;
}

// CHECK-LABEL:   func.func @f3(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xf64>,
// CHECK-SAME:                  %[[VAL_1:.*]]: f32) -> f64
// CHECK:           %[[VAL_2:.*]] = arith.extf %[[VAL_1]] : f32 to f64
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xf64>
// CHECK:           %[[VAL_4:.*]] = arith.mulf %[[VAL_3]], %[[VAL_2]] : f64
// CHECK:           affine.store %[[VAL_4]], %[[VAL_0]][0] : memref<?xf64>
// CHECK:           return %[[VAL_4]] : f64
// CHECK:         }

double f3(double *a, float b) {
  return *a *= b;
}

// CHECK-LABEL:   func.func @f4(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xf64>,
// CHECK-SAME:                  %[[VAL_1:.*]]: i32) -> f64
// CHECK:           %[[VAL_2:.*]] = arith.sitofp %[[VAL_1]] : i32 to f64
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xf64>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_3]], %[[VAL_2]] : f64
// CHECK:           affine.store %[[VAL_4]], %[[VAL_0]][0] : memref<?xf64>
// CHECK:           return %[[VAL_4]] : f64
// CHECK:         }

double f4(double *a, int b) {
  return *a += b;
}

// CHECK-LABEL:   func.func @f5(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: f64) -> i32
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_3:.*]] = arith.sitofp %[[VAL_2]] : i32 to f64
// CHECK:           %[[VAL_4:.*]] = arith.divf %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_5:.*]] = arith.fptosi %[[VAL_4]] : f64 to i32
// CHECK:           affine.store %[[VAL_5]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }

int f5(int *a, double b) {
  return *a /= b;
}

// CHECK-LABEL:   func.func @f6(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xf64>,
// CHECK-SAME:                  %[[VAL_1:.*]]: i32) -> f64
// CHECK:           %[[VAL_2:.*]] = arith.uitofp %[[VAL_1]] : i32 to f64
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xf64>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_3]], %[[VAL_2]] : f64
// CHECK:           affine.store %[[VAL_4]], %[[VAL_0]][0] : memref<?xf64>
// CHECK:           return %[[VAL_4]] : f64
// CHECK:         }

double f6(double *a, unsigned b) {
  return *a += b;
}

// CHECK-LABEL:   func.func @f7(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: f64) -> i32
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_3:.*]] = arith.uitofp %[[VAL_2]] : i32 to f64
// CHECK:           %[[VAL_4:.*]] = arith.divf %[[VAL_3]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_5:.*]] = arith.fptoui %[[VAL_4]] : f64 to i32
// CHECK:           affine.store %[[VAL_5]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }

unsigned f7(unsigned *a, double b) {
  return *a /= b;
}

// CHECK-LABEL:   func.func @f8(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: i64) -> i32
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = arith.ori %[[VAL_3]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i64 to i32
// CHECK:           affine.store %[[VAL_5]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }

unsigned f8(unsigned *a, unsigned long b) {
  return *a |= b;
}

// CHECK-LABEL:   func.func @f9(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xi64>,
// CHECK-SAME:                  %[[VAL_1:.*]]: i32) -> i64
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_1]] : i32 to i64
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xi64>
// CHECK:           %[[VAL_4:.*]] = arith.andi %[[VAL_3]], %[[VAL_2]] : i64
// CHECK:           affine.store %[[VAL_4]], %[[VAL_0]][0] : memref<?xi64>
// CHECK:           return %[[VAL_4]] : i64
// CHECK:         }

unsigned long f9(unsigned long *a, int b) {
  return *a &= b;
}

// CHECK-LABEL:   func.func @f10(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_3:.*]] = arith.shli %[[VAL_2]], %[[VAL_1]] : i32
// CHECK:           affine.store %[[VAL_3]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }

unsigned f10(unsigned *a, unsigned b) {
  return *a <<= b;
}

// CHECK-LABEL:   func.func @f11(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_3:.*]] = arith.shrui %[[VAL_2]], %[[VAL_1]] : i32
// CHECK:           affine.store %[[VAL_3]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }

unsigned f11(unsigned *a, unsigned b) {
  return *a >>= b;
}

// CHECK-LABEL:   func.func @f12(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_3:.*]] = arith.xori %[[VAL_2]], %[[VAL_1]] : i32
// CHECK:           affine.store %[[VAL_3]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }

unsigned f12(unsigned *a, unsigned b) {
  return *a ^= b;
}

// CHECK-LABEL:   func.func @f13(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi8>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i32) -> i1
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xi8>
// CHECK:           %[[VAL_4:.*]] = arith.extui %[[VAL_3]] : i8 to i32
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_7:.*]] = arith.extui %[[VAL_6]] : i1 to i8
// CHECK:           affine.store %[[VAL_7]], %[[VAL_0]][0] : memref<?xi8>
// CHECK:           return %[[VAL_6]] : i1
// CHECK:         }

bool f13(bool *a, unsigned b) {
  return *a += b;
}

// CHECK-LABEL:   func.func @f14(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi8>,
// CHECK-SAME:                   %[[VAL_1:.*]]: f32) -> i1
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xi8>
// CHECK:           %[[VAL_4:.*]] = arith.uitofp %[[VAL_3]] : i8 to f32
// CHECK:           %[[VAL_5:.*]] = arith.addf %[[VAL_4]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_6:.*]] = arith.cmpf une, %[[VAL_5]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_7:.*]] = arith.extui %[[VAL_6]] : i1 to i8
// CHECK:           affine.store %[[VAL_7]], %[[VAL_0]][0] : memref<?xi8>
// CHECK:           return %[[VAL_6]] : i1
// CHECK:         }

bool f14(bool *a, float b) {
  return *a += b;
}

// CHECK-LABEL:   func.func @f15(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi8>,
// CHECK-SAME:                   %[[VAL_1:.*]]: f64) -> i1
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xi8>
// CHECK:           %[[VAL_4:.*]] = arith.uitofp %[[VAL_3]] : i8 to f64
// CHECK:           %[[VAL_5:.*]] = arith.addf %[[VAL_4]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_6:.*]] = arith.cmpf une, %[[VAL_5]], %[[VAL_2]] : f64
// CHECK:           %[[VAL_7:.*]] = arith.extui %[[VAL_6]] : i1 to i8
// CHECK:           affine.store %[[VAL_7]], %[[VAL_0]][0] : memref<?xi8>
// CHECK:           return %[[VAL_6]] : i1
// CHECK:         }

bool f15(bool *a, double b) {
  return *a += b;
}
