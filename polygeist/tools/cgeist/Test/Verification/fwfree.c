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

// CHECK:     func @main() -> i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %alloc = memref.alloc() : memref<2800xi32>
// CHECK-NEXT:     %cast = memref.cast %alloc : memref<2800xi32> to memref<?xi32>
// CHECK-NEXT:     call @init_array(%cast) : (memref<?xi32>) -> ()
// CHECK-NEXT:     memref.dealloc %alloc : memref<2800xi32>
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }

// FULLRANK: %[[MEM:.*]] = memref.alloc() : memref<2800xi32>
// FULLRANK: call @init_array(%[[MEM]]) : (memref<2800xi32>) -> ()
