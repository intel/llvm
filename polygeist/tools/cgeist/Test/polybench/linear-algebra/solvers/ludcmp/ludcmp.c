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
/* ludcmp.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "ludcmp.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(b,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      x[i] = 0;
      y[i] = 0;
      b[i] = (i+1)/fn/2.0 + 4;
    }

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
  }
  POLYBENCH_DUMP_END("x");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_ludcmp(int n,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(b,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j, k;

  DATA_TYPE w;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <i; j++) {
       w = A[i][j];
       for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   for (j = i; j < _PB_N; j++) {
       w = A[i][j];
       for (k = 0; k < i; k++) {
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  for (i = 0; i < _PB_N; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   for (i = _PB_N-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < _PB_N; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(b),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_ludcmp (n,
		 POLYBENCH_ARRAY(A),
		 POLYBENCH_ARRAY(b),
		 POLYBENCH_ARRAY(x),
		 POLYBENCH_ARRAY(y));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(b);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}

// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK-NEXT: #map1 = affine_map<(d0)[s0] -> (-d0 + s0)>
// CHECK:   func @kernel_ludcmp(%arg0: i32, %arg1: memref<2000x2000xf64>, %arg2: memref<2000xf64>, %arg3: memref<2000xf64>, %arg4: memref<2000xf64>) {
// CHECK-NEXT:      %c0 = constant 0 : index
// CHECK-NEXT:      %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:      %1 = alloca() : memref<1xf64>
// CHECK-NEXT:      %2 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %3 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %4 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %5 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      affine.for %arg5 = 0 to %0 {
// CHECK-NEXT:        affine.for %arg6 = 0 to #map0(%arg5) {
// CHECK-NEXT:          %10 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          affine.store %10, %1[0] : memref<1xf64>
// CHECK-NEXT:          affine.for %arg7 = 0 to #map0(%arg6) {
// CHECK-NEXT:            %13 = affine.load %arg1[%arg5, %arg7] : memref<2000x2000xf64>
// CHECK-NEXT:            %14 = affine.load %arg1[%arg7, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:            %15 = mulf %13, %14 : f64
// CHECK-NEXT:            %16 = subf %2, %15 : f64
// CHECK-NEXT:            affine.store %16, %1[0] : memref<1xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:          %11 = affine.load %arg1[%arg6, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          %12 = divf %3, %11 : f64
// CHECK-NEXT:          affine.store %12, %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %arg6 = #map0(%arg5) to %0 {
// CHECK-NEXT:          %10 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          affine.store %10, %1[0] : memref<1xf64>
// CHECK-NEXT:          affine.for %arg7 = 0 to #map0(%arg5) {
// CHECK-NEXT:            %11 = affine.load %arg1[%arg5, %arg7] : memref<2000x2000xf64>
// CHECK-NEXT:            %12 = affine.load %arg1[%arg7, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:            %13 = mulf %11, %12 : f64
// CHECK-NEXT:            %14 = subf %4, %13 : f64
// CHECK-NEXT:            affine.store %14, %1[0] : memref<1xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:          affine.store %5, %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %6 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %7 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      affine.for %arg5 = 0 to %0 {
// CHECK-NEXT:        %10 = affine.load %arg2[%arg5] : memref<2000xf64>
// CHECK-NEXT:        affine.store %10, %1[0] : memref<1xf64>
// CHECK-NEXT:        affine.for %arg6 = 0 to #map0(%arg5) {
// CHECK-NEXT:          %11 = affine.load %arg1[%arg5, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          %12 = affine.load %arg4[%arg6] : memref<2000xf64>
// CHECK-NEXT:          %13 = mulf %11, %12 : f64
// CHECK-NEXT:          %14 = subf %6, %13 : f64
// CHECK-NEXT:          affine.store %14, %1[0] : memref<1xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.store %7, %arg4[%arg5] : memref<2000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %8 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      %9 = load %1[%c0] : memref<1xf64>
// CHECK-NEXT:      affine.for %arg5 = 0 to %0 {
// CHECK-NEXT:        %10 = affine.load %arg4[-%arg5 + symbol(%0) - 1] : memref<2000xf64>
// CHECK-NEXT:        affine.store %10, %1[0] : memref<1xf64>
// CHECK-NEXT:        affine.for %arg6 = #map1(%arg5)[%0] to %0 {
// CHECK-NEXT:          %13 = affine.load %arg1[-%arg5 + symbol(%0) - 1, %arg6] : memref<2000x2000xf64>
// CHECK-NEXT:          %14 = affine.load %arg3[%arg6] : memref<2000xf64>
// CHECK-NEXT:          %15 = mulf %13, %14 : f64
// CHECK-NEXT:          %16 = subf %8, %15 : f64
// CHECK-NEXT:          affine.store %16, %1[0] : memref<1xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %11 = affine.load %arg1[-%arg5 + symbol(%0) - 1, -%arg5 + symbol(%0) - 1] : memref<2000x2000xf64>
// CHECK-NEXT:        %12 = divf %9, %11 : f64
// CHECK-NEXT:        affine.store %12, %arg3[-%arg5 + symbol(%0) - 1] : memref<2000xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// EXEC: {{[0-9]\.[0-9]+}}
