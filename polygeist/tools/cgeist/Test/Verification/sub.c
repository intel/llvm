// RUN: cgeist %s --function=* -S | FileCheck %s

#include <stddef.h>

typedef int int_vec __attribute__((ext_vector_type(3)));
typedef float float_vec __attribute__((ext_vector_type(3)));

// CHECK-LABEL:   func.func @f0(
// CHECK-SAME:                  %[[VAL_0:.*]]: i32,
// CHECK-SAME:                  %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[SUB:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[SUB]] : i32
// CHECK:         }

int f0(int a, int b) {
  return a - b;
}

// CHECK-LABEL:   func.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: f32,
// CHECK-SAME:                  %[[VAL_1:.*]]: f32) -> f32
// CHECK:           %[[SUB:.*]] = arith.subf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           return %[[SUB]] : f32
// CHECK:         }

float f1(float a, float b) {
  return a - b;
}

// CHECK-LABEL:   func.func @f2(
// CHECK-SAME:                  %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK:           %[[SUB:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK:           return %[[SUB]] : vector<3xi32>
// CHECK:         }

int_vec f2(int_vec a, int_vec b) {
  return a - b;
}

// CHECK-LABEL:   func.func @f3(
// CHECK-SAME:                  %[[VAL_0:.*]]: vector<3xf32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: vector<3xf32>) -> vector<3xf32>
// CHECK:           %[[SUB:.*]] = arith.subf %[[VAL_0]], %[[VAL_1]] : vector<3xf32>
// CHECK:           return %[[SUB]] : vector<3xf32>
// CHECK:         }

float_vec f3(float_vec a, float_vec b) {
  return a - b;
}

// CHECK-LABEL:   func.func @f4(
// CHECK-SAME:                  %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                  %[[VAL_1:.*]]: !llvm.ptr<i8>) -> i64
// CHECK:           %[[INT_0:.*]] = llvm.ptrtoint %[[VAL_0]] : !llvm.ptr<i8> to i64
// CHECK:           %[[INT_1:.*]] = llvm.ptrtoint %[[VAL_1]] : !llvm.ptr<i8> to i64
// CHECK:           %[[SUB:.*]] = arith.subi %[[INT_0]], %[[INT_1]] : i64
// CHECK:           return %[[SUB]] : i64
// CHECK:         }

size_t f4(char *a, char *b) {
  return a - b;
}

// CHECK-LABEL:   func.func @f5(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xf32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: memref<?xf32>) -> i64
// CHECK:           %[[I64_0:.*]] = arith.constant 4 : i64
// CHECK:           %[[PTR_0:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?xf32>) -> !llvm.ptr<f32>
// CHECK:           %[[INT_0:.*]] = llvm.ptrtoint %[[PTR_0]] : !llvm.ptr<f32> to i64
// CHECK:           %[[PTR_1:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?xf32>) -> !llvm.ptr<f32>
// CHECK:           %[[INT_1:.*]] = llvm.ptrtoint %[[PTR_1]] : !llvm.ptr<f32> to i64
// CHECK:           %[[DIFF:.*]] = arith.subi %[[INT_0]], %[[INT_1]] : i64
// CHECK:           %[[SUB:.*]] = arith.divsi %[[DIFF]], %[[I64_0]] : i64
// CHECK:           return %[[SUB]] : i64
// CHECK:         }

size_t f5(float *a, float *b) {
  return a - b;
}
