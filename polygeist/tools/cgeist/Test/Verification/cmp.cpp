// RUN: cgeist %s -O0 -w --function=* -S | FileCheck %s

struct Compare {
  bool lt;
  bool gt;
  bool le;
  bool ge;
  bool eq;
  bool ne;
};

template <typename T>
Compare compare(T lhs, T rhs) {
  return {lhs < rhs, lhs > rhs, lhs <= rhs, lhs >= rhs, lhs == rhs, lhs != rhs};
}

// CHECK-LABEL:   func.func @_Z7compareIfE7CompareT_S1_(
// CHECK-SAME:                                          %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32)
// CHECK:           %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_8:.*]] = arith.cmpf ogt, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_11:.*]] = arith.cmpf ole, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_14:.*]] = arith.cmpf oge, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_17:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_20:.*]] = arith.cmpf une, %[[VAL_0]], %[[VAL_1]] : f32
template Compare compare(float lhs, float rhs);

// CHECK-LABEL:   func.func @_Z7compareIdE7CompareT_S1_(
// CHECK-SAME:                                          %[[VAL_0:.*]]: f64, %[[VAL_1:.*]]: f64)
// CHECK:           %[[VAL_5:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_8:.*]] = arith.cmpf ogt, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_11:.*]] = arith.cmpf ole, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_14:.*]] = arith.cmpf oge, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_17:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK:           %[[VAL_20:.*]] = arith.cmpf une, %[[VAL_0]], %[[VAL_1]] : f64
template Compare compare(double lhs, double rhs);
