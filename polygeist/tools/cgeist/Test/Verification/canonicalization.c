// RUN: cgeist --S --function=* --memref-fullrank %s | FileCheck %s

// The following should be able to fully lower to memref ops without memref
// subviews.

// CHECK-LABEL:   func @matrix_power(
// CHECK:                       %[[VAL_0:.*]]: memref<20x20xi32>, %[[VAL_1:.*]]: memref<20xi32>, %[[VAL_2:.*]]: memref<20xi32>, %[[VAL_3:.*]]: memref<20xi32>)
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c20 = arith.constant 20 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     scf.for %arg4 = %c1 to %c20 step %c1 {
// CHECK-NEXT:       %0 = arith.index_cast %arg4 : index to i32
// CHECK-NEXT:       %1 = arith.addi %0, %c-1_i32 : i32
// CHECK-NEXT:       %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:       scf.for %arg5 = %c0 to %c20 step %c1 {
// CHECK-NEXT:         %3 = memref.load %[[VAL_3]][%arg5] : memref<20xi32>
// CHECK-NEXT:         %4 = memref.load %[[VAL_2]][%arg5] : memref<20xi32>
// CHECK-NEXT:         %5 = arith.index_cast %4 : i32 to index
// CHECK-NEXT:         %6 = memref.load %[[VAL_0]][%2, %5] : memref<20x20xi32>
// CHECK-NEXT:         %7 = arith.muli %3, %6 : i32
// CHECK-NEXT:         %8 = memref.load %[[VAL_1]][%arg5] : memref<20xi32>
// CHECK-NEXT:         %9 = arith.index_cast %8 : i32 to index
// CHECK-NEXT:         %10 = memref.load %[[VAL_0]][%arg4, %9] : memref<20x20xi32>
// CHECK-NEXT:         %11 = arith.addi %10, %7 : i32
// CHECK-NEXT:         memref.store %11, %arg0[%arg4, %9] : memref<20x20xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }     
// CHECK-NEXT:     return
// CHECK-NEXT:   }

void matrix_power(int x[20][20], int row[20], int col[20], int a[20]) {
  for (int k = 1; k < 20; k++) {
    for (int p = 0; p < 20; p++) {
      x[k][row[p]] += a[p] * x[k - 1][col[p]];
    }
  }
}
