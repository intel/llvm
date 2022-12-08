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

// CHECK-LABEL:   func.func @test_store(
// CHECK-SAME:                          %[[VAL_0:.*]]: memref<?xvector<3xi32>>,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32,
// CHECK-SAME:                          %[[VAL_2:.*]]: i32) -> i32 attributes
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi32>>
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.insertelement %[[VAL_2]], %[[VAL_3]]{{\[}}%[[VAL_1]] : i32] : vector<3xi32>
// CHECK-NEXT:      affine.store %[[VAL_4]], %[[VAL_0]][0] : memref<?xvector<3xi32>>
// CHECK-NEXT:      return %[[VAL_2]] : i32
// CHECK-NEXT:    }

int test_store(int_vec *v, int idx, int el) {
  return (*v)[idx] = el;
}

// CHECK-LABEL:   func.func @test_inc(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xi32>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: i32,
// CHECK-SAME:                        %[[VAL_2:.*]]: i32) -> i32
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi32>>
// CHECK-NEXT:      %[[VAL_4:.*]] = vector.extractelement %[[VAL_3]]{{\[}}%[[VAL_1]] : i32] : vector<3xi32>
// CHECK-NEXT:      %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_2]] : i32
// CHECK-NEXT:      %[[VAL_6:.*]] = vector.insertelement %[[VAL_5]], %[[VAL_3]]{{\[}}%[[VAL_1]] : i32] : vector<3xi32>
// CHECK-NEXT:      affine.store %[[VAL_6]], %[[VAL_0]][0] : memref<?xvector<3xi32>>
// CHECK-NEXT:      return %[[VAL_5]] : i32
// CHECK-NEXT:    }

int test_inc(int_vec *v, int idx, int el) {
  return (*v)[idx] += el;
}

// CHECK-LABEL:   func.func @test_gen(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) -> vector<3xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_7:.*]] = llvm.mlir.undef : vector<3xi32>
// CHECK-NEXT:      %[[VAL_8:.*]] = vector.insertelement %[[VAL_0]], %[[VAL_7]]{{\[}}%[[VAL_5]] : i32] : vector<3xi32>
// CHECK-NEXT:      %[[VAL_9:.*]] = vector.insertelement %[[VAL_1]], %[[VAL_8]]{{\[}}%[[VAL_4]] : i32] : vector<3xi32>
// CHECK-NEXT:      %[[VAL_10:.*]] = vector.insertelement %[[VAL_2]], %[[VAL_9]]{{\[}}%[[VAL_3]] : i32] : vector<3xi32>
// CHECK-NEXT:      return %[[VAL_10]] : vector<3xi32>
// CHECK-NEXT:    }

int_vec test_gen(int a, int b, int c) {
  int_vec v;
  v[0] = a;
  v[1] = b;
  v[2] = c;
  return v;
}
