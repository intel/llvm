// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

typedef char char_vec __attribute__((ext_vector_type(3)));
typedef short short_vec __attribute__((ext_vector_type(3)));
typedef int int_vec __attribute__((ext_vector_type(3)));
typedef long long_vec __attribute__((ext_vector_type(3)));
typedef float float_vec __attribute__((ext_vector_type(3)));
typedef double double_vec __attribute__((ext_vector_type(3)));

// CHECK-LABEL:   func.func @minus_i8(
// CHECK-SAME:                        %[[VAL_0:.*]]: i8) -> i8
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i8 to i32
// CHECK:           %[[VAL_3:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i32 to i8
// CHECK:           return %[[VAL_4]] : i8
// CHECK:         }

char minus_i8(char a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_i16(
// CHECK-SAME:                         %[[VAL_0:.*]]: i16) -> i16
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i16 to i32
// CHECK:           %[[VAL_3:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i32 to i16
// CHECK:           return %[[VAL_4]] : i16
// CHECK:         }

short minus_i16(short a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_i32(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32) -> i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : i32
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }

int minus_i32(int a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_i64(
// CHECK-SAME:                         %[[VAL_0:.*]]: i64) -> i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : i64
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         }

long minus_i64(long a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_f32(
// CHECK-SAME:                         %[[VAL_0:.*]]: f32) -> f32
// CHECK:           %[[VAL_1:.*]] = arith.negf %[[VAL_0]] : f32
// CHECK:           return %[[VAL_1]] : f32
// CHECK:         }

float minus_f32(float a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_f64(
// CHECK-SAME:                         %[[VAL_0:.*]]: f64) -> f64
// CHECK:           %[[VAL_1:.*]] = arith.negf %[[VAL_0]] : f64
// CHECK:           return %[[VAL_1]] : f64
// CHECK:         }

double minus_f64(double a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_vi8(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<3xi8>) -> vector<3xi8>
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xi8>
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : vector<3xi8>
// CHECK:           return %[[VAL_2]] : vector<3xi8>
// CHECK:         }

char_vec minus_vi8(char_vec a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_vf16(
// CHECK-SAME:                          %[[VAL_0:.*]]: vector<3xi16>) -> vector<3xi16>
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xi16>
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : vector<3xi16>
// CHECK:           return %[[VAL_2]] : vector<3xi16>
// CHECK:         }

short_vec minus_vf16(short_vec a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_vi32(
// CHECK-SAME:                          %[[VAL_0:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xi32>
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : vector<3xi32>
// CHECK:           return %[[VAL_2]] : vector<3xi32>
// CHECK:         }

int_vec minus_vi32(int_vec a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_vi64(
// CHECK-SAME:                          %[[VAL_0:.*]]: memref<?xvector<3xi64>>) -> vector<3xi64>
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0> : vector<3xi64>
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi64>>
// CHECK:           %[[VAL_3:.*]] = arith.subi %[[VAL_1]], %[[VAL_2]] : vector<3xi64>
// CHECK:           return %[[VAL_3]] : vector<3xi64>
// CHECK:         }

long_vec minus_vi64(long_vec a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_vf32(
// CHECK-SAME:                          %[[VAL_0:.*]]: vector<3xf32>) -> vector<3xf32>
// CHECK:           %[[VAL_1:.*]] = arith.negf %[[VAL_0]] : vector<3xf32>
// CHECK:           return %[[VAL_1]] : vector<3xf32>
// CHECK:         }

float_vec minus_vf32(float_vec a) {
  return -a;
}

// CHECK-LABEL:   func.func @minus_vf64(
// CHECK-SAME:                          %[[VAL_0:.*]]: memref<?xvector<3xf64>>) -> vector<3xf64>
// CHECK:           %[[VAL_1:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xf64>>
// CHECK:           %[[VAL_2:.*]] = arith.negf %[[VAL_1]] : vector<3xf64>
// CHECK:           return %[[VAL_2]] : vector<3xf64>
// CHECK:         }

double_vec minus_vf64(double_vec a) {
  return -a;
}
