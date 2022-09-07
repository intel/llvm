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
/* heat-3d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "heat-3d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))

{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
         if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
         fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_heat_3d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int t, i, j, k;

#pragma scop
    for (t = 1; t <= TSTEPS; t++) {
        for (i = 1; i < _PB_N-1; i++) {
            for (j = 1; j < _PB_N-1; j++) {
                for (k = 1; k < _PB_N-1; k++) {
                    B[i][j][k] =   SCALAR_VAL(0.125) * (A[i+1][j][k] - SCALAR_VAL(2.0) * A[i][j][k] + A[i-1][j][k])
                                 + SCALAR_VAL(0.125) * (A[i][j+1][k] - SCALAR_VAL(2.0) * A[i][j][k] + A[i][j-1][k])
                                 + SCALAR_VAL(0.125) * (A[i][j][k+1] - SCALAR_VAL(2.0) * A[i][j][k] + A[i][j][k-1])
                                 + A[i][j][k];
                }
            }
        }
        for (i = 1; i < _PB_N-1; i++) {
           for (j = 1; j < _PB_N-1; j++) {
               for (k = 1; k < _PB_N-1; k++) {
                   A[i][j][k] =   SCALAR_VAL(0.125) * (B[i+1][j][k] - SCALAR_VAL(2.0) * B[i][j][k] + B[i-1][j][k])
                                + SCALAR_VAL(0.125) * (B[i][j+1][k] - SCALAR_VAL(2.0) * B[i][j][k] + B[i][j-1][k])
                                + SCALAR_VAL(0.125) * (B[i][j][k+1] - SCALAR_VAL(2.0) * B[i][j][k] + B[i][j][k-1])
                                + B[i][j][k];
               }
           }
       }
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_heat_3d (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}

// CHECK: #map = affine_map<()[s0] -> (s0 - 1)>

// CHECK:   func private @kernel_heat_3d(%arg0: i32, %arg1: i32, %arg2: memref<120x120x120xf64>, %arg3: memref<120x120x120xf64>) {
// CHECK-NEXT:    %cst = constant 1.250000e-01 : f64
// CHECK-NEXT:    %cst_0 = constant 2.000000e+00 : f64
// CHECK-NEXT:    %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:    affine.for %arg4 = 1 to 501 {
// CHECK-NEXT:      affine.for %arg5 = 1 to #map()[%0] {
// CHECK-NEXT:        affine.for %arg6 = 1 to #map()[%0] {
// CHECK-NEXT:          affine.for %arg7 = 1 to #map()[%0] {
// CHECK-NEXT:            %1 = affine.load %arg2[%arg5 + 1, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %2 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %3 = mulf %cst_0, %2 : f64
// CHECK-NEXT:            %4 = subf %1, %3 : f64
// CHECK-NEXT:            %5 = affine.load %arg2[%arg5 - 1, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %6 = addf %4, %5 : f64
// CHECK-NEXT:            %7 = mulf %cst, %6 : f64
// CHECK-NEXT:            %8 = affine.load %arg2[%arg5, %arg6 + 1, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %9 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %10 = mulf %cst_0, %9 : f64
// CHECK-NEXT:            %11 = subf %8, %10 : f64
// CHECK-NEXT:            %12 = affine.load %arg2[%arg5, %arg6 - 1, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %13 = addf %11, %12 : f64
// CHECK-NEXT:            %14 = mulf %cst, %13 : f64
// CHECK-NEXT:            %15 = addf %7, %14 : f64
// CHECK-NEXT:            %16 = affine.load %arg2[%arg5, %arg6, %arg7 + 1] : memref<120x120x120xf64>
// CHECK-NEXT:            %17 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %18 = mulf %cst_0, %17 : f64
// CHECK-NEXT:            %19 = subf %16, %18 : f64
// CHECK-NEXT:            %20 = affine.load %arg2[%arg5, %arg6, %arg7 - 1] : memref<120x120x120xf64>
// CHECK-NEXT:            %21 = addf %19, %20 : f64
// CHECK-NEXT:            %22 = mulf %cst, %21 : f64
// CHECK-NEXT:            %23 = addf %15, %22 : f64
// CHECK-NEXT:            %24 = affine.load %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %25 = addf %23, %24 : f64
// CHECK-NEXT:            affine.store %25, %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg5 = 1 to #map()[%0] {
// CHECK-NEXT:        affine.for %arg6 = 1 to #map()[%0] {
// CHECK-NEXT:          affine.for %arg7 = 1 to #map()[%0] {
// CHECK-NEXT:            %1 = affine.load %arg3[%arg5 + 1, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %2 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %3 = mulf %cst_0, %2 : f64
// CHECK-NEXT:            %4 = subf %1, %3 : f64
// CHECK-NEXT:            %5 = affine.load %arg3[%arg5 - 1, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %6 = addf %4, %5 : f64
// CHECK-NEXT:            %7 = mulf %cst, %6 : f64
// CHECK-NEXT:            %8 = affine.load %arg3[%arg5, %arg6 + 1, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %9 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %10 = mulf %cst_0, %9 : f64
// CHECK-NEXT:            %11 = subf %8, %10 : f64
// CHECK-NEXT:            %12 = affine.load %arg3[%arg5, %arg6 - 1, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %13 = addf %11, %12 : f64
// CHECK-NEXT:            %14 = mulf %cst, %13 : f64
// CHECK-NEXT:            %15 = addf %7, %14 : f64
// CHECK-NEXT:            %16 = affine.load %arg3[%arg5, %arg6, %arg7 + 1] : memref<120x120x120xf64>
// CHECK-NEXT:            %17 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %18 = mulf %cst_0, %17 : f64
// CHECK-NEXT:            %19 = subf %16, %18 : f64
// CHECK-NEXT:            %20 = affine.load %arg3[%arg5, %arg6, %arg7 - 1] : memref<120x120x120xf64>
// CHECK-NEXT:            %21 = addf %19, %20 : f64
// CHECK-NEXT:            %22 = mulf %cst, %21 : f64
// CHECK-NEXT:            %23 = addf %15, %22 : f64
// CHECK-NEXT:            %24 = affine.load %arg3[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:            %25 = addf %23, %24 : f64
// CHECK-NEXT:            affine.store %25, %arg2[%arg5, %arg6, %arg7] : memref<120x120x120xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// EXEC: {{[0-9]\.[0-9]+}}
