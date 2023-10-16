// RUN: cgeist %s %stdinclude --function=init_array -S | FileCheck %s
// RUN: cgeist %s %stdinclude --function=init_array -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#include <stdio.h>
#include <unistd.h>
#include <string.h>

/* Include polybench common header. */
#include <polybench.h>

void use(double A[20]);
/* Array initialization. */

void init_array (int n)
{
  double (*B)[20] = (double(*)[20])polybench_alloc_data (20, sizeof(double)) ;
  (*B)[2] = 3.0;
  use(*B);
}


// CHECK-LABEL:   func.func @init_array(
// CHECK-SAME:                          %[[VAL_0:.*]]: i32)
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 8 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.constant 20 : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = call @polybench_alloc_data(%[[VAL_3]], %[[VAL_1]]) : (i64, i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_4]][2] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK-NEXT:      llvm.store %[[VAL_2]], %[[VAL_5]] : f64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = "polygeist.pointer2memref"(%[[VAL_4]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:      call @use(%[[VAL_6]]) : (memref<?xf64>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// FULLRANK: %[[VAL0:.*]] = "polygeist.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<20xf64>
// FULLRANK: call @use(%[[VAL0]]) : (memref<20xf64>) -> ()
