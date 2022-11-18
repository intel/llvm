// RUN: cgeist %s -O2 %stdinclude -S | FileCheck %s
// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: cgeist %s -O3 %polyverify %stdinclude -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: cgeist %s -O3 %polyexec %stdinclude -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm %s.mlir.time %s.clang.time

// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: cgeist %s -O3 %polyverify %stdinclude -detect-reduction -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2

/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
      {
	A[i] = ((DATA_TYPE) i+ 2) / n;
	B[i] = ((DATA_TYPE) i+ 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
//static
void kernel_jacobi_1d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
    {
      for (i = 1; i < _PB_N - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < _PB_N - 1; i++)
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}

// CHECK: [[MAP:#map.*]] = affine_map<()[s0] -> (s0 - 1)>
// CHECK: func @kernel_jacobi_1d(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<?xf64>)
// CHECK-NEXT:      %cst = arith.constant 3.333300e-01 : f64
// CHECK-NEXT:      %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:      %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:      affine.for %arg4 = 0 to %1 {
// CHECK-NEXT:        affine.for %arg5 = 1 to [[MAP]]()[%0] {
// CHECK-NEXT:          %2 = affine.load %arg2[%arg5 - 1] : memref<?xf64>
// CHECK-NEXT:          %3 = affine.load %arg2[%arg5] : memref<?xf64>
// CHECK-NEXT:          %4 = arith.addf %2, %3 : f64
// CHECK-NEXT:          %5 = affine.load %arg2[%arg5 + 1] : memref<?xf64>
// CHECK-NEXT:          %6 = arith.addf %4, %5 : f64
// CHECK-NEXT:          %7 = arith.mulf %6, %cst : f64
// CHECK-NEXT:          affine.store %7, %arg3[%arg5] : memref<?xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %arg5 = 1 to [[MAP]]()[%0] {
// CHECK-NEXT:          %2 = affine.load %arg3[%arg5 - 1] : memref<?xf64>
// CHECK-NEXT:          %3 = affine.load %arg3[%arg5] : memref<?xf64>
// CHECK-NEXT:          %4 = arith.addf %2, %3 : f64
// CHECK-NEXT:          %5 = affine.load %arg3[%arg5 + 1] : memref<?xf64>
// CHECK-NEXT:          %6 = arith.addf %4, %5 : f64
// CHECK-NEXT:          %7 = arith.mulf %6, %cst : f64
// CHECK-NEXT:          affine.store %7, %arg2[%arg5] : memref<?xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// EXEC: {{[0-9]\.[0-9]+}}
