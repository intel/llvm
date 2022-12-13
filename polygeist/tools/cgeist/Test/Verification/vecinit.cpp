// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));

// CHECK-LABEL:   func.func @_Z15test_constant_1v() -> vector<8xf32>
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : vector<8xf32>
// CHECK-NEXT:      return %[[VAL_0]] : vector<8xf32>
// CHECK-NEXT:    }
float8 test_constant_1() {
  return {0, 1, 2, 3, 4, 5, 6, 7};
}

// CHECK-LABEL:   func.func @_Z15test_constant_2v() -> vector<8xf32>
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : vector<8xf32>
// CHECK-NEXT:      return %[[VAL_0]] : vector<8xf32>
// CHECK-NEXT:    }
float8 test_constant_2() {
  auto t = float8{0, 1, 2, 3, 4, 5, 6, 7};
  return t;
}

// CHECK-LABEL:   func.func @_Z15test_constant_3v() -> vector<8xf32>
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : vector<8xf32>
// CHECK-NEXT:      return %[[VAL_0]] : vector<8xf32>
// CHECK-NEXT:    }
float8 test_constant_3() {
  float8 t{0, 1, 2, 3, 4, 5, 6, 7};
  return t;
}

// CHECK-LABEL:   func.func @_Z9test_eachffffffff(
// CHECK-SAME:                                    %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f32) -> vector<8xf32>
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.insert %[[VAL_0]], %[[VAL_8]] [0] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_10:.*]] = vector.insert %[[VAL_1]], %[[VAL_9]] [1] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_11:.*]] = vector.insert %[[VAL_2]], %[[VAL_10]] [2] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_12:.*]] = vector.insert %[[VAL_3]], %[[VAL_11]] [3] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_13:.*]] = vector.insert %[[VAL_4]], %[[VAL_12]] [4] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_14:.*]] = vector.insert %[[VAL_5]], %[[VAL_13]] [5] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_15:.*]] = vector.insert %[[VAL_6]], %[[VAL_14]] [6] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_16:.*]] = vector.insert %[[VAL_7]], %[[VAL_15]] [7] : f32 into vector<8xf32>
// CHECK-NEXT:      return %[[VAL_16]] : vector<8xf32>
// CHECK-NEXT:    }
float8 test_each(float A, float B, float C, float D,
		 float E, float F, float G, float H) {
  return {A, B, C, D, E, F, G, H};
}

// CHECK-LABEL:   func.func @_Z9test_copyDv8_f(
// CHECK-SAME:                                 %[[VAL_0:.*]]: memref<?xvector<8xf32>>) -> vector<8xf32>
// CHECK-NEXT:      %[[VAL_1:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<8xf32>>
// CHECK-NEXT:      return %[[VAL_1]] : vector<8xf32>
// CHECK-NEXT:    }
float8 test_copy(float8 Arg0) {
  return {Arg0};
}

// CHECK-LABEL:   func.func @_Z9test_fillffff(
// CHECK-SAME:                                %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32) -> vector<8xf32>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-NEXT:      %[[VAL_5:.*]] = vector.insert %[[VAL_0]], %[[VAL_4]] [0] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.insert %[[VAL_1]], %[[VAL_5]] [1] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_7:.*]] = vector.insert %[[VAL_2]], %[[VAL_6]] [2] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.insert %[[VAL_3]], %[[VAL_7]] [3] : f32 into vector<8xf32>
// CHECK-NEXT:      return %[[VAL_8]] : vector<8xf32>
// CHECK-NEXT:    }
float8 test_fill(float A, float B, float C, float D) {
  return {A, B, C, D};
}

// CHECK-LABEL:   func.func @_Z11test_concatDv4_fS_(
// CHECK-SAME:                                      %[[VAL_0:.*]]: vector<4xf32>, %[[VAL_1:.*]]: vector<4xf32>) -> vector<8xf32>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-NEXT:      %[[VAL_3:.*]] = vector.extract %[[VAL_0]][0] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.insert %[[VAL_3]], %[[VAL_2]] [0] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_5:.*]] = vector.extract %[[VAL_0]][1] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.insert %[[VAL_5]], %[[VAL_4]] [1] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_7:.*]] = vector.extract %[[VAL_0]][2] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.insert %[[VAL_7]], %[[VAL_6]] [2] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.extract %[[VAL_0]][3] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_10:.*]] = vector.insert %[[VAL_9]], %[[VAL_8]] [3] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_11:.*]] = vector.extract %[[VAL_1]][0] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_12:.*]] = vector.insert %[[VAL_11]], %[[VAL_10]] [4] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_13:.*]] = vector.extract %[[VAL_1]][1] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_14:.*]] = vector.insert %[[VAL_13]], %[[VAL_12]] [5] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_15:.*]] = vector.extract %[[VAL_1]][2] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_16:.*]] = vector.insert %[[VAL_15]], %[[VAL_14]] [6] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_17:.*]] = vector.extract %[[VAL_1]][3] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_18:.*]] = vector.insert %[[VAL_17]], %[[VAL_16]] [7] : f32 into vector<8xf32>
// CHECK-NEXT:      return %[[VAL_18]] : vector<8xf32>
// CHECK-NEXT:    }
float8 test_concat(float4 A, float4 B) {
  return {A.x, A.y, A.z, A.w, B.x, B.y, B.z, B.w};
}

// CHECK-LABEL:   func.func @_Z11test_expandDv4_f(
// CHECK-SAME:                                    %[[VAL_0:.*]]: vector<4xf32>) -> vector<8xf32>
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-NEXT:      %[[VAL_2:.*]] = vector.extract %[[VAL_0]][0] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_3:.*]] = vector.insert %[[VAL_2]], %[[VAL_1]] [0] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.extract %[[VAL_0]][1] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_5:.*]] = vector.insert %[[VAL_4]], %[[VAL_3]] [1] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.extract %[[VAL_0]][2] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_7:.*]] = vector.insert %[[VAL_6]], %[[VAL_5]] [2] : f32 into vector<8xf32>
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.extract %[[VAL_0]][3] : vector<4xf32>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.insert %[[VAL_8]], %[[VAL_7]] [3] : f32 into vector<8xf32>
// CHECK-NEXT:      return %[[VAL_9]] : vector<8xf32>
// CHECK-NEXT:    }
float8 test_expand(float4 A) {
  return {A.x, A.y, A.z, A.w};
}
