// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

#include <stddef.h>

typedef char char_vec __attribute__((ext_vector_type(3)));
typedef short short_vec __attribute__((ext_vector_type(3)));
typedef int int_vec __attribute__((ext_vector_type(3)));
typedef long long_vec __attribute__((ext_vector_type(3)));
typedef float float_vec __attribute__((ext_vector_type(3)));
typedef double double_vec __attribute__((ext_vector_type(3)));

// CHECK-LABEL:   func.func @sub_i8(
// CHECK-SAME:                      %[[VAL_0:.*]]: i8,
// CHECK-SAME:                      %[[VAL_1:.*]]: i8) -> i8
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i8 to i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_1]] : i8 to i32
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i8
// CHECK:           return %[[VAL_5]] : i8
// CHECK:         }

char sub_i8(char a, char b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_i16(
// CHECK-SAME:                       %[[VAL_0:.*]]: i16,
// CHECK-SAME:                       %[[VAL_1:.*]]: i16) -> i16
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i16 to i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_1]] : i16 to i32
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i16
// CHECK:           return %[[VAL_5]] : i16
// CHECK:         }

short sub_i16(short a, short b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_i32(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32,
// CHECK-SAME:                       %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }

int sub_i32(int a, int b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_i64(
// CHECK-SAME:                       %[[VAL_0:.*]]: i64,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64) -> i64
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         }

long sub_i64(long a, long b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_f32(
// CHECK-SAME:                  %[[VAL_0:.*]]: f32,
// CHECK-SAME:                  %[[VAL_1:.*]]: f32) -> f32
// CHECK:           %[[SUB:.*]] = arith.subf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           return %[[SUB]] : f32
// CHECK:         }

float sub_f32(float a, float b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_f64(
// CHECK-SAME:                       %[[VAL_0:.*]]: f32,
// CHECK-SAME:                       %[[VAL_1:.*]]: f32) -> f32
// CHECK:           %[[VAL_2:.*]] = arith.subf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           return %[[VAL_2]] : f32
// CHECK:         }

float sub_f64(float a, float b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_vi8(
// CHECK-SAME:                       %[[VAL_0:.*]]: vector<3xi8>,
// CHECK-SAME:                       %[[VAL_1:.*]]: vector<3xi8>) -> vector<3xi8>
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : vector<3xi8>
// CHECK:           return %[[VAL_2]] : vector<3xi8>
// CHECK:         }

char_vec sub_vi8(char_vec a, char_vec b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_vi16(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi16>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi16>) -> vector<3xi16>
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : vector<3xi16>
// CHECK:           return %[[VAL_2]] : vector<3xi16>
// CHECK:         }

short_vec sub_vi16(short_vec a, short_vec b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_vi32(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK:           return %[[VAL_2]] : vector<3xi32>
// CHECK:         }

int_vec sub_vi32(int_vec a, int_vec b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_vi64(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xi64>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xvector<3xi64>>) -> vector<3xi64>
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi64>>
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xi64>>
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : vector<3xi64>
// CHECK:           return %[[VAL_4]] : vector<3xi64>
// CHECK:         }

long_vec sub_vi64(long_vec a, long_vec b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_vf32(
// CHECK-SAME:                  %[[VAL_0:.*]]: vector<3xf32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: vector<3xf32>) -> vector<3xf32>
// CHECK:           %[[SUB:.*]] = arith.subf %[[VAL_0]], %[[VAL_1]] : vector<3xf32>
// CHECK:           return %[[SUB]] : vector<3xf32>
// CHECK:         }

float_vec sub_vf32(float_vec a, float_vec b) {
  return a - b;
}

// CHECK-LABEL:   func.func @sub_vf64(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xf64>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xvector<3xf64>>) -> vector<3xf64>
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xf64>>
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xf64>>
// CHECK:           %[[VAL_4:.*]] = arith.subf %[[VAL_2]], %[[VAL_3]] : vector<3xf64>
// CHECK:           return %[[VAL_4]] : vector<3xf64>
// CHECK:         }

double_vec sub_vf64(double_vec a, double_vec b) {
  return a - b;
}

// CHECK-LABEL:   func.func @ptr_diff_i8(
// CHECK-SAME:                  %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-SAME:                  %[[VAL_1:.*]]: !llvm.ptr<i8>) -> i64
// CHECK:           %[[VAL_2:.*]] = llvm.ptrtoint %[[VAL_0]] : !llvm.ptr<i8> to i64
// CHECK:           %[[VAL_3:.*]] = llvm.ptrtoint %[[VAL_1]] : !llvm.ptr<i8> to i64
// CHECK:           %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : i64
// CHECK:           return %[[VAL_4]] : i64
// CHECK:         }

size_t ptr_diff_i8(char *a, char *b) {
  return a - b;
}

// CHECK-LABEL:   func.func @ptr_diff_f32(
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

size_t ptr_diff_f32(float *a, float *b) {
  return a - b;
}
