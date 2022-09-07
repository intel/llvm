// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: cgeist %s %polyverify %stdinclude -O3 -o %s.execm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: cgeist %s %polyexec %stdinclude -O3 -o %s.execm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm %s.mlir.time %s.clang.time

// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
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
/* atax.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_atax(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    y[i] = 0;
  for (i = 0; i < _PB_M; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < _PB_N; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}

// CHECK:   func @kernel_atax(%arg0: i32, %arg1: i32, %arg2: memref<1900x2100xf64>, %arg3: memref<2100xf64>, %arg4: memref<2100xf64>, %arg5: memref<1900xf64>) {
// CHECK-NEXT:  %c0_i32 = constant 0 : i32
// CHECK-NEXT:  %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:  %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:  %1 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:  affine.for %arg6 = 0 to %0 {
// CHECK-NEXT:    affine.store %1, %arg4[%arg6] : memref<2100xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = index_cast %arg0 : i32 to index
// CHECK-NEXT:  affine.for %arg6 = 0 to %2 {
// CHECK-NEXT:    affine.store %cst, %arg5[%arg6] : memref<1900xf64>
// CHECK-NEXT:    %3 = affine.load %arg5[%arg6] : memref<1900xf64>
// CHECK-NEXT:    affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:      %5 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
// CHECK-NEXT:      %6 = affine.load %arg3[%arg7] : memref<2100xf64>
// CHECK-NEXT:      %7 = mulf %5, %6 : f64
// CHECK-NEXT:      %8 = addf %3, %7 : f64
// CHECK-NEXT:      affine.store %8, %arg5[%arg6] : memref<1900xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = affine.load %arg5[%arg6] : memref<1900xf64>
// CHECK-NEXT:    affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:      %5 = affine.load %arg4[%arg7] : memref<2100xf64>
// CHECK-NEXT:      %6 = affine.load %arg2[%arg6, %arg7] : memref<1900x2100xf64>
// CHECK-NEXT:      %7 = mulf %6, %4 : f64
// CHECK-NEXT:      %8 = addf %5, %7 : f64
// CHECK-NEXT:      affine.store %8, %arg4[%arg7] : memref<2100xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}

// EXEC: {{[0-9]\.[0-9]+}}
