// RUN: cgeist %s %stdinclude -S | FileCheck %s
// RUN: cgeist %s %stdinclude -S -memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

#   define N 2800

/* Array initialization. */
void init_array (int path[N]);

int main()
{
  /* Retrieve problem size. */

  /* Variable declaration/allocation. */
  //POLYBENCH_1D_ARRAY_DECL(path, int, N, n);
  int (*path)[N];
  //int path[POLYBENCH_C99_SELECT(N,n) + POLYBENCH_PADDING_FACTOR];
  path = (int(*)[N])polybench_alloc_data (N, sizeof(int)) ;

  /* Initialize array(s). */
  init_array (*path);

  POLYBENCH_FREE_ARRAY(path);
  return 0;
}

// CHECK-LABEL:   func.func @main() -> i32
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 4 : i32
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 2800 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = call @polybench_alloc_data(%[[VAL_2]], %[[VAL_0]]) : (i64, i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = "polygeist.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<2800xi32>
// CHECK-NEXT:      %[[VAL_5:.*]] = "polygeist.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:      call @init_array(%[[VAL_5]]) : (memref<?xi32>) -> ()
// CHECK-NEXT:      memref.dealloc %[[VAL_4]] : memref<2800xi32>
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }

// FULLRANK: %[[MEM:.*]] = "polygeist.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<2800xi32>
// FULLRANK: call @init_array(%[[MEM]]) : (memref<2800xi32>) -> ()
