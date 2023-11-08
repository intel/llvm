// RUN: cgeist -w -o - -S --function=* %s | FileCheck %s

using double2 = double __attribute__((ext_vector_type(2)));

template <typename T>
T test_simple_lhs(T a, T b, T c) {
  return a * b + c;
}

template <typename T>
T test_negmul_lhs(T a, T b, T c) {
  return -(a * b) + c;
}

template <typename T>
T test_negadd_lhs(T a, T b, T c) {
  return a * b - c;
}

template <typename T>
T test_negmul_negadd_lhs(T a, T b, T c) {
  return -(a * b) - c;
}

template <typename T>
T test_simple_rhs(T a, T b, T c) {
  return a + b * c;
}

template <typename T>
T test_negmul_rhs(T a, T b, T c) {
  return a + -(b * c);
}

template <typename T>
T test_negadd_rhs(T a, T b, T c) {
  return a - b * c;
}

template <typename T>
T test_negmul_negadd_rhs(T a, T b, T c) {
  return a - -(b * c);
}

#define TEST_TYPE(type)                                         \
  template type test_simple_lhs(type a, type b, type c);        \
  template type test_negmul_lhs(type a, type b, type c);        \
  template type test_negadd_lhs(type a, type b, type c);        \
  template type test_negmul_negadd_lhs(type a, type b, type c); \
  template type test_simple_rhs(type a, type b, type c);        \
  template type test_negmul_rhs(type a, type b, type c);        \
  template type test_negadd_rhs(type a, type b, type c);        \
  template type test_negmul_negadd_rhs(type a, type b, type c);

TEST_TYPE(float)

// CHECK-LABEL:   func.func @_Z15test_simple_lhsIfET_S0_S0_S0_(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32,
// CHECK-SAME:                                                 %[[VAL_2:.*]]: f32) -> f32
// CHECK:           %[[VAL_3:.*]] = math.fma %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : f32
// CHECK:           return %[[VAL_3]] : f32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_negmul_lhsIfET_S0_S0_S0_(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32) -> f32
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_0]] : f32
// CHECK:           %[[VAL_4:.*]] = math.fma %[[VAL_3]], %[[VAL_1]], %[[VAL_2]] : f32
// CHECK:           return %[[VAL_4]] : f32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_negadd_lhsIfET_S0_S0_S0_(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32) -> f32
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_2]] : f32
// CHECK:           %[[VAL_4:.*]] = math.fma %[[VAL_0]], %[[VAL_1]], %[[VAL_3]] : f32
// CHECK:           return %[[VAL_4]] : f32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z22test_negmul_negadd_lhsIfET_S0_S0_S0_(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32) -> f32
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_0]] : f32
// CHECK:           %[[VAL_4:.*]] = arith.negf %[[VAL_2]] : f32
// CHECK:           %[[VAL_5:.*]] = math.fma %[[VAL_3]], %[[VAL_1]], %[[VAL_4]] : f32
// CHECK:           return %[[VAL_5]] : f32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_simple_rhsIfET_S0_S0_S0_(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32) -> f32
// CHECK:           %[[VAL_3:.*]] = math.fma %[[VAL_1]], %[[VAL_2]], %[[VAL_0]] : f32
// CHECK:           return %[[VAL_3]] : f32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_negmul_rhsIfET_S0_S0_S0_(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32) -> f32
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_1]] : f32
// CHECK:           %[[VAL_4:.*]] = math.fma %[[VAL_3]], %[[VAL_2]], %[[VAL_0]] : f32
// CHECK:           return %[[VAL_4]] : f32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_negadd_rhsIfET_S0_S0_S0_(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32) -> f32
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_1]] : f32
// CHECK:           %[[VAL_4:.*]] = math.fma %[[VAL_3]], %[[VAL_2]], %[[VAL_0]] : f32
// CHECK:           return %[[VAL_4]] : f32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z22test_negmul_negadd_rhsIfET_S0_S0_S0_(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32) -> f32
// CHECK:           %[[VAL_3:.*]] = math.fma %[[VAL_1]], %[[VAL_2]], %[[VAL_0]] : f32
// CHECK:           return %[[VAL_3]] : f32
// CHECK:         }

TEST_TYPE(double2)

// CHECK-LABEL:   func.func @_Z15test_simple_lhsIDv2_dET_S1_S1_S1_(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: vector<2xf64>, %[[VAL_1:.*]]: vector<2xf64>, %[[VAL_2:.*]]: vector<2xf64>) -> vector<2xf64>
// CHECK:           %[[VAL_3:.*]] = math.fma %[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : vector<2xf64>
// CHECK:           return %[[VAL_3]] : vector<2xf64>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_negmul_lhsIDv2_dET_S1_S1_S1_(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: vector<2xf64>, %[[VAL_1:.*]]: vector<2xf64>, %[[VAL_2:.*]]: vector<2xf64>) -> vector<2xf64>
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_0]] : vector<2xf64>
// CHECK:           %[[VAL_4:.*]] = math.fma %[[VAL_3]], %[[VAL_1]], %[[VAL_2]] : vector<2xf64>
// CHECK:           return %[[VAL_4]] : vector<2xf64>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_negadd_lhsIDv2_dET_S1_S1_S1_(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: vector<2xf64>, %[[VAL_1:.*]]: vector<2xf64>, %[[VAL_2:.*]]: vector<2xf64>) -> vector<2xf64>
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_2]] : vector<2xf64>
// CHECK:           %[[VAL_4:.*]] = math.fma %[[VAL_0]], %[[VAL_1]], %[[VAL_3]] : vector<2xf64>
// CHECK:           return %[[VAL_4]] : vector<2xf64>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z22test_negmul_negadd_lhsIDv2_dET_S1_S1_S1_(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: vector<2xf64>, %[[VAL_1:.*]]: vector<2xf64>, %[[VAL_2:.*]]: vector<2xf64>) -> vector<2xf64>
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_0]] : vector<2xf64>
// CHECK:           %[[VAL_4:.*]] = arith.negf %[[VAL_2]] : vector<2xf64>
// CHECK:           %[[VAL_5:.*]] = math.fma %[[VAL_3]], %[[VAL_1]], %[[VAL_4]] : vector<2xf64>
// CHECK:           return %[[VAL_5]] : vector<2xf64>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_simple_rhsIDv2_dET_S1_S1_S1_(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: vector<2xf64>, %[[VAL_1:.*]]: vector<2xf64>, %[[VAL_2:.*]]: vector<2xf64>) -> vector<2xf64>
// CHECK:           %[[VAL_3:.*]] = math.fma %[[VAL_1]], %[[VAL_2]], %[[VAL_0]] : vector<2xf64>
// CHECK:           return %[[VAL_3]] : vector<2xf64>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_negmul_rhsIDv2_dET_S1_S1_S1_(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: vector<2xf64>, %[[VAL_1:.*]]: vector<2xf64>, %[[VAL_2:.*]]: vector<2xf64>) -> vector<2xf64>
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_1]] : vector<2xf64>
// CHECK:           %[[VAL_4:.*]] = math.fma %[[VAL_3]], %[[VAL_2]], %[[VAL_0]] : vector<2xf64>
// CHECK:           return %[[VAL_4]] : vector<2xf64>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z15test_negadd_rhsIDv2_dET_S1_S1_S1_(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: vector<2xf64>, %[[VAL_1:.*]]: vector<2xf64>, %[[VAL_2:.*]]: vector<2xf64>) -> vector<2xf64>
// CHECK:           %[[VAL_3:.*]] = arith.negf %[[VAL_1]] : vector<2xf64>
// CHECK:           %[[VAL_4:.*]] = math.fma %[[VAL_3]], %[[VAL_2]], %[[VAL_0]] : vector<2xf64>
// CHECK:           return %[[VAL_4]] : vector<2xf64>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z22test_negmul_negadd_rhsIDv2_dET_S1_S1_S1_(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: vector<2xf64>, %[[VAL_1:.*]]: vector<2xf64>, %[[VAL_2:.*]]: vector<2xf64>) -> vector<2xf64>
// CHECK:           %[[VAL_3:.*]] = math.fma %[[VAL_1]], %[[VAL_2]], %[[VAL_0]] : vector<2xf64>
// CHECK:           return %[[VAL_3]] : vector<2xf64>
// CHECK:         }
