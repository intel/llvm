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
/* covariance.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)n;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < _PB_M; i++)
    for (j = i; j < _PB_M; j++)
      {
        cov[i][j] = SCALAR_VAL(0.0);
        for (k = 0; k < _PB_N; k++)
	  cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - SCALAR_VAL(1.0));
        cov[j][i] = cov[i][j];
      }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(cov,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);


  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance (m, n, float_n,
		     POLYBENCH_ARRAY(data),
		     POLYBENCH_ARRAY(cov),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>) {
// CHECK-NEXT:  %cst = constant 0.000000e+00 : f64
// CHECK-NEXT:  %cst_0 = constant 1.000000e+00 : f64
// CHECK-NEXT:  %0 = index_cast %arg0 : i32 to index
// CHECK-NEXT:  %1 = index_cast %arg1 : i32 to index
// CHECK-NEXT:  affine.for %arg6 = 0 to %0 {
// CHECK-NEXT:    affine.store %cst, %arg5[%arg6] : memref<1200xf64>
// CHECK-NEXT:      %3 = affine.load %arg5[%arg6] : memref<1200xf64>
// CHECK-NEXT:      affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:        %6 = affine.load %arg3[%arg7, %arg6] : memref<1400x1200xf64>
// CHECK-NEXT:        %7 = addf %3, %6 : f64
// CHECK-NEXT:        affine.store %7, %arg5[%arg6] : memref<1200xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %4 = affine.load %arg5[%arg6] : memref<1200xf64>
// CHECK-NEXT:      %5 = divf %4, %arg2 : f64
// CHECK-NEXT:      affine.store %5, %arg5[%arg6] : memref<1200xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  affine.for %arg6 = 0 to %1 {
// CHECK-NEXT:    affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:      %3 = affine.load %arg5[%arg7] : memref<1200xf64>
// CHECK-NEXT:      %4 = affine.load %arg3[%arg6, %arg7] : memref<1400x1200xf64>
// CHECK-NEXT:      %5 = subf %4, %3 : f64
// CHECK-NEXT:      affine.store %5, %arg3[%arg6, %arg7] : memref<1400x1200xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %2 = subf %arg2, %cst_0 : f64
// CHECK-NEXT:  affine.for %arg6 = 0 to %0 {
// CHECK-NEXT:    affine.for %arg7 = #map(%arg6) to %0 {
// CHECK-NEXT:      affine.store %cst, %arg4[%arg6, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:        %3 = affine.load %arg4[%arg6, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:        affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:          %7 = affine.load %arg3[%arg8, %arg6] : memref<1400x1200xf64>
// CHECK-NEXT:          %8 = affine.load %arg3[%arg8, %arg7] : memref<1400x1200xf64>
// CHECK-NEXT:          %9 = mulf %7, %8 : f64
// CHECK-NEXT:          %10 = addf %3, %9 : f64
// CHECK-NEXT:          affine.store %10, %arg4[%arg6, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        %4 = affine.load %arg4[%arg6, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:        %5 = divf %4, %2 : f64
// CHECK-NEXT:        affine.store %5, %arg4[%arg6, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:        %6 = affine.load %arg4[%arg6, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:        affine.store %6, %arg4[%arg7, %arg6] : memref<1200x1200xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}

// EXEC: {{[0-9]\.[0-9]+}}
