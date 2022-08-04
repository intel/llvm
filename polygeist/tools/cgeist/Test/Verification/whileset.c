// RUN: cgeist %s %stdinclude --function=set -S | FileCheck %s
// RUN: cgeist %s %stdinclude --function=set -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>


/* Array initialization. */

void set (int path[20])
{
    int i = 0;
    while (1) {
        path[i] = 3;
        i++;
        if (i == 20) break;
    }
  //path[0][1] = 2;
}

// TODO consider making into for
// CHECK:       func @set(%arg0: memref<?xi32>)
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-DAG:     %c20_i32 = arith.constant 20 : i32
// CHECK-DAG:     %true = arith.constant true
// CHECK-NEXT:     %0 = scf.while (%arg1 = %c0_i32, %arg2 = %true) : (i32, i1) -> i32 {
// CHECK-NEXT:       scf.condition(%arg2) %arg1 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg1: i32):
// CHECK-NEXT:       %1 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:       memref.store %c3_i32, %arg0[%1] : memref<?xi32>
// CHECK-NEXT:       %2 = arith.addi %arg1, %c1_i32 : i32
// CHECK-NEXT:       %3 = arith.cmpi ne, %2, %c20_i32 : i32
// CHECK-NEXT:       scf.yield %2, %3 : i32, i1
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// FULLRANK: func @set(%{{.*}}: memref<20xi32>)
