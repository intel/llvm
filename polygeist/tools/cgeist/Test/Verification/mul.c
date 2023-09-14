// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

typedef char char_vec __attribute__((ext_vector_type(3)));
typedef short short_vec __attribute__((ext_vector_type(3)));
typedef int int_vec __attribute__((ext_vector_type(3)));
typedef long long_vec __attribute__((ext_vector_type(3)));
typedef float float_vec __attribute__((ext_vector_type(3)));
typedef double double_vec __attribute__((ext_vector_type(3)));


// CHECK-LABEL:   func.func @mul_i8(
// CHECK-SAME:                      %[[VAL_0:.*]]: i8,
// CHECK-SAME:                      %[[VAL_1:.*]]: i8) -> i8
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i8 to i32
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.extsi %[[VAL_1]] : i8 to i32
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.muli %[[VAL_2]], %[[VAL_3]] : i32
// CHECK-NEXT:      %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i8
// CHECK-NEXT:      return %[[VAL_5]] : i8
// CHECK-NEXT:    }
char mul_i8(char a, char b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_i16(
// CHECK-SAME:                       %[[VAL_0:.*]]: i16,
// CHECK-SAME:                       %[[VAL_1:.*]]: i16) -> i16
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i16 to i32
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.extsi %[[VAL_1]] : i16 to i32
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.muli %[[VAL_2]], %[[VAL_3]] : i32
// CHECK-NEXT:      %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i16
// CHECK-NEXT:      return %[[VAL_5]] : i16
// CHECK-NEXT:    }
short mul_i16(short a, short b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_i32(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32,
// CHECK-SAME:                       %[[VAL_1:.*]]: i32) -> i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK-NEXT:      return %[[VAL_2]] : i32
// CHECK-NEXT:    }
int mul_i32(int a, int b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_i64(
// CHECK-SAME:                       %[[VAL_0:.*]]: i64,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64) -> i64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : i64
// CHECK-NEXT:      return %[[VAL_2]] : i64
// CHECK-NEXT:    }
long mul_i64(long a, long b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_f32(
// CHECK-SAME:                       %[[VAL_0:.*]]: f32,
// CHECK-SAME:                       %[[VAL_1:.*]]: f32) -> f32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.mulf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK-NEXT:      return %[[VAL_2]] : f32
// CHECK-NEXT:    }
float mul_f32(float a, float b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_f64(
// CHECK-SAME:                       %[[VAL_0:.*]]: f32,
// CHECK-SAME:                       %[[VAL_1:.*]]: f32) -> f32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.mulf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK-NEXT:      return %[[VAL_2]] : f32
// CHECK-NEXT:    }
float mul_f64(float a, float b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_vi8(
// CHECK-SAME:                       %[[VAL_0:.*]]: vector<3xi8>,
// CHECK-SAME:                       %[[VAL_1:.*]]: vector<3xi8>) -> vector<3xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : vector<3xi8>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi8>
// CHECK-NEXT:    }
char_vec mul_vi8(char_vec a, char_vec b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_vi16(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi16>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi16>) -> vector<3xi16>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : vector<3xi16>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi16>
// CHECK-NEXT:    }
short_vec mul_vi16(short_vec a, short_vec b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_vi32(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi32>
// CHECK-NEXT:    }
int_vec mul_vi32(int_vec a, int_vec b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_vi64(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xi64>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xvector<3xi64>>) -> vector<3xi64>
// CHECK-NEXT:      %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.muli %[[VAL_2]], %[[VAL_3]] : vector<3xi64>
// CHECK-NEXT:      return %[[VAL_4]] : vector<3xi64>
// CHECK-NEXT:    }
long_vec mul_vi64(long_vec a, long_vec b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_vf32(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xf32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xf32>) -> vector<3xf32>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.mulf %[[VAL_0]], %[[VAL_1]] : vector<3xf32>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xf32>
// CHECK-NEXT:    }
float_vec mul_vf32(float_vec a, float_vec b) {
  return a * b;
}


// CHECK-LABEL:   func.func @mul_vf64(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xf64>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xvector<3xf64>>) -> vector<3xf64>
// CHECK-NEXT:      %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xf64>>
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xf64>>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.mulf %[[VAL_2]], %[[VAL_3]] : vector<3xf64>
// CHECK-NEXT:      return %[[VAL_4]] : vector<3xf64>
// CHECK-NEXT:    }
double_vec mul_vf64(double_vec a, double_vec b) {
  return a * b;
}
