// RUN: cgeist %s --function=* -S | FileCheck %s

typedef int int_vec __attribute__((ext_vector_type(3)));
typedef float float_vec __attribute__((ext_vector_type(3)));

// CHECK-LABEL:   func.func @f0(
// CHECK-SAME:                  %[[VAL_0:.*]]: i32,
// CHECK-SAME:                  %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[ADD:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[ADD]] : i32
// CHECK:         }

int f0(int a, int b) {
  return a + b;
}

// CHECK-LABEL:   func.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: f32,
// CHECK-SAME:                  %[[VAL_1:.*]]: f32) -> f32
// CHECK:           %[[ADD:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           return %[[ADD]] : f32
// CHECK:         }

float f1(float a, float b) {
  return a + b;
}

// CHECK-LABEL:   func.func @f2(
// CHECK-SAME:                  %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK:           %[[ADD:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK:           return %[[ADD]] : vector<3xi32>
// CHECK:         }

int_vec f2(int_vec a, int_vec b) {
  return a + b;
}

// CHECK-LABEL:   func.func @f4(
// CHECK-SAME:                  %[[VAL_0:.*]]: vector<3xf32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: vector<3xf32>) -> vector<3xf32>
// CHECK:           %[[ADD:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : vector<3xf32>
// CHECK:           return %[[ADD]] : vector<3xf32>
// CHECK:         }

float_vec f4(float_vec a, float_vec b) {
  return a + b;
}
