// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 -lm && %s.exec1 &> %s.out1
// RUN: cgeist %s %polyverify %stdinclude -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: cgeist %s %polyexec %stdinclude -O3 -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 -lm && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm %s.mlir.time %s.clang.time

// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 -lm && %s.exec1 &> %s.out1
// RUN: cgeist %s %polyverify %stdinclude -detect-reduction -O3 -o %s.execm && %s.execm &> %s.out2
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
/* gramschmidt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* QR Decomposition with Modified Gram Schmidt:
 http://www.inf.ethz.ch/personal/gander/ */
static
void kernel_gramschmidt(int m, int n,
			DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
			DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;

  DATA_TYPE nrm;

  for (k = 0; k < _PB_N; k++)
    {
      nrm = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_M; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = SQRT_FUN(nrm);
      for (i = 0; i < _PB_M; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < _PB_N; j++)
	{
	  R[k][j] = SCALAR_VAL(0.0);
	  for (i = 0; i < _PB_M; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  for (i = 0; i < _PB_M; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt (m, n,
		      POLYBENCH_ARRAY(A),
		      POLYBENCH_ARRAY(R),
		      POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}

// CHECK: #map = affine_map<(d0) -> (d0 + 1)>
// CHECK:  func @kernel_gramschmidt(%arg0: i32, %arg1: i32, %arg2: memref<1000x1200xf64>, %arg3: memref<1200x1200xf64>, %arg4: memref<1000x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = index_cast %arg1 : i32 to index
// CHECK-NEXT:     store %cst, %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %2 = index_cast %arg0 : i32 to index
// CHECK-NEXT:     %3 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %4 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     %5 = sqrt %4 : f64
// CHECK-NEXT:     affine.for %arg5 = 0 to %1 {
// CHECK-NEXT:       affine.for %arg6 = 0 to %2 {
// CHECK-NEXT:         %7 = affine.load %arg2[%arg6, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:         %8 = affine.load %arg2[%arg6, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:         %9 = mulf %7, %8 : f64
// CHECK-NEXT:         %10 = addf %3, %9 : f64
// CHECK-NEXT:         affine.store %10, %0[0] : memref<1xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %5, %arg3[%arg5, %arg5] : memref<1200x1200xf64>
// CHECK-NEXT:       %6 = affine.load %arg3[%arg5, %arg5] : memref<1200x1200xf64>
// CHECK-NEXT:       affine.for %arg6 = 0 to %2 {
// CHECK-NEXT:         %7 = affine.load %arg2[%arg6, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:         %8 = divf %7, %6 : f64
// CHECK-NEXT:         affine.store %8, %arg4[%arg6, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.for %arg6 = #map(%arg5) to %1 {
// CHECK-NEXT:         affine.store %cst, %arg3[%arg5, %arg6] : memref<1200x1200xf64>
// CHECK-NEXT:         %7 = affine.load %arg3[%arg5, %arg6] : memref<1200x1200xf64>
// CHECK-NEXT:         affine.for %arg7 = 0 to %2 {
// CHECK-NEXT:           %9 = affine.load %arg4[%arg7, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:           %10 = affine.load %arg2[%arg7, %arg6] : memref<1000x1200xf64>
// CHECK-NEXT:           %11 = mulf %9, %10 : f64
// CHECK-NEXT:           %12 = addf %7, %11 : f64
// CHECK-NEXT:           affine.store %12, %arg3[%arg5, %arg6] : memref<1200x1200xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         %8 = affine.load %arg3[%arg5, %arg6] : memref<1200x1200xf64>
// CHECK-NEXT:         affine.for %arg7 = 0 to %2 {
// CHECK-NEXT:           %9 = affine.load %arg2[%arg7, %arg6] : memref<1000x1200xf64>
// CHECK-NEXT:           %10 = affine.load %arg4[%arg7, %arg5] : memref<1000x1200xf64>
// CHECK-NEXT:           %11 = mulf %10, %8 : f64
// CHECK-NEXT:           %12 = subf %9, %11 : f64
// CHECK-NEXT:           affine.store %12, %arg2[%arg7, %arg6] : memref<1000x1200xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// EXEC: {{[0-9]\.[0-9]+}}
