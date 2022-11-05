// RUN: cgeist %s --function=* -S -o - | FileCheck %s

// COM: We need to check this in C++ as it's illegal in C.

// CHECK-LABEL:   func.func @{{.*}}f1{{.*}}(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32,
// CHECK-SAME:                         %[[VAL_2:.*]]: i32) -> i32
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_2]] : i32
// CHECK:           affine.store %[[VAL_5]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }

int f1(int &a, int b, int c) {
  return (a += b) += c;
}

// CHECK-LABEL:   func.func @{{.*}}f2{{.*}}(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32,
// CHECK-SAME:                         %[[VAL_2:.*]]: i32) -> i32
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : i32
// CHECK:           affine.store %[[VAL_5]], %[[VAL_0]][0] : memref<?xi32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }

int f2(int &a, int b, int c) {
  return a += (b += c);
}
