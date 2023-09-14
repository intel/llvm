// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

typedef char char_vec __attribute__((ext_vector_type(3)));
typedef short short_vec __attribute__((ext_vector_type(3)));
typedef int int_vec __attribute__((ext_vector_type(3)));
typedef long long_vec __attribute__((ext_vector_type(3)));
typedef float float_vec __attribute__((ext_vector_type(3)));
typedef double double_vec __attribute__((ext_vector_type(3)));

// CHECK-LABEL:   func.func @add_i8(
// CHECK-SAME:                      %[[VAL_0:.*]]: i8,
// CHECK-SAME:                      %[[VAL_1:.*]]: i8) -> i8
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i8 to i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_1]] : i8 to i32
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i8
// CHECK:           return %[[VAL_5]] : i8
// CHECK:         }

char add_i8(char a, char b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_i16(
// CHECK-SAME:                       %[[VAL_0:.*]]: i16,
// CHECK-SAME:                       %[[VAL_1:.*]]: i16) -> i16
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i16 to i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_1]] : i16 to i32
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i16
// CHECK:           return %[[VAL_5]] : i16
// CHECK:         }

short add_i16(short a, short b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_i32(
// CHECK-SAME:                  %[[VAL_0:.*]]: i32,
// CHECK-SAME:                  %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[ADD:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[ADD]] : i32
// CHECK:         }

int add_i32(int a, int b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_i64(
// CHECK-SAME:                       %[[VAL_0:.*]]: i64,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64) -> i64
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i64
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         }

long add_i64(long a, long b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_f32(
// CHECK-SAME:                  %[[VAL_0:.*]]: f32,
// CHECK-SAME:                  %[[VAL_1:.*]]: f32) -> f32
// CHECK:           %[[ADD:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           return %[[ADD]] : f32
// CHECK:         }

float add_f32(float a, float b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_f64(
// CHECK-SAME:                       %[[VAL_0:.*]]: f32,
// CHECK-SAME:                       %[[VAL_1:.*]]: f32) -> f32
// CHECK:           %[[VAL_2:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           return %[[VAL_2]] : f32
// CHECK:         }

float add_f64(float a, float b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_vi8(
// CHECK-SAME:                  %[[VAL_0:.*]]: vector<3xi8>,
// CHECK-SAME:                  %[[VAL_1:.*]]: vector<3xi8>) -> vector<3xi8>
// CHECK:           %[[ADD:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : vector<3xi8>
// CHECK:           return %[[ADD]] : vector<3xi8>
// CHECK:         }

char_vec add_vi8(char_vec a, char_vec b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_vi16(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi16>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi16>) -> vector<3xi16>
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : vector<3xi16>
// CHECK:           return %[[VAL_2]] : vector<3xi16>
// CHECK:         }

short_vec add_vi16(short_vec a, short_vec b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_vi32(
// CHECK-SAME:                  %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK:           %[[ADD:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK:           return %[[ADD]] : vector<3xi32>
// CHECK:         }

int_vec add_vi32(int_vec a, int_vec b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_vi64(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xi64>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xvector<3xi64>>) -> vector<3xi64>
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi64>>
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xi64>>
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : vector<3xi64>
// CHECK:           return %[[VAL_4]] : vector<3xi64>
// CHECK:         }

long_vec add_vi64(long_vec a, long_vec b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_vf32(
// CHECK-SAME:                  %[[VAL_0:.*]]: vector<3xf32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: vector<3xf32>) -> vector<3xf32>
// CHECK:           %[[ADD:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : vector<3xf32>
// CHECK:           return %[[ADD]] : vector<3xf32>
// CHECK:         }

float_vec add_vf32(float_vec a, float_vec b) {
  return a + b;
}

// CHECK-LABEL:   func.func @add_vf64(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xf64>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xvector<3xf64>>) -> vector<3xf64>
// CHECK:           %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xf64>>
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xf64>>
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : vector<3xf64>
// CHECK:           return %[[VAL_4]] : vector<3xf64>
// CHECK:         }

double_vec add_vf64(double_vec a, double_vec b) {
  return a + b;
}
