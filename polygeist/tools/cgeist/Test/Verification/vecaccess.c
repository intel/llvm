// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

typedef int int_vec __attribute__((ext_vector_type(3)));

// CHECK-LABEL:   func.func @test_cons_idx(
// CHECK-SAME:                             %[[VAL_0:.*]]: vector<3xi32>) -> i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = vector.extractelement %[[VAL_0]]{{\[}}%[[VAL_1]] : i32] : vector<3xi32>
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }
int test_cons_idx(int_vec v) {
  return v[0];
}

// CHECK-LABEL:   func.func @test_var_idx(
// CHECK-SAME:                    %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i32
// CHECK:           %[[VAL_2:.*]] = vector.extractelement %[[VAL_0]]{{\[}}%[[VAL_1]] : i32] : vector<3xi32>
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }
int test_var_idx(int_vec v, unsigned i) {
  return v[i];
}
