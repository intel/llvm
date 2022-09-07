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
/* adi.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static
void kernel_adi(int tsteps, int n,
		DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

#pragma scop

  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;

 for (t=1; t<=_PB_TSTEPS; t++) {
    //Column Sweep
    for (i=1; i<_PB_N-1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -c / (a*p[i][j-1]+b);
        q[i][j] = (-d*u[j][i-1]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b);
      }

      v[_PB_N-1][i] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
    }
    //Row Sweep
    for (i=1; i<_PB_N-1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -f / (d*p[i][j-1]+e);
        q[i][j] = (-a*v[i-1][j]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e);
      }
      u[i][_PB_N-1] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
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
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

  return 0;
}

// CHECK: #map0 = affine_map<()[s0] -> (s0 + 1)>
// CHECK: #map1 = affine_map<()[s0] -> (s0 - 1)>

// CHECK:    func private @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000x1000xf64>) {
// CHECK-NEXT:     %cst = constant 1.000000e+00 : f64
// CHECK-NEXT:     %cst_0 = constant 2.000000e+00 : f64
// CHECK-NEXT:     %cst_1 = constant 0.000000e+00 : f64
// CHECK-NEXT:     %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = sitofp %arg1 : i32 to f64
// CHECK-NEXT:     %2 = divf %cst, %1 : f64
// CHECK-NEXT:     %3 = sitofp %arg0 : i32 to f64
// CHECK-NEXT:     %4 = divf %cst, %3 : f64
// CHECK-NEXT:     %5 = mulf %cst_0, %4 : f64
// CHECK-NEXT:     %6 = mulf %2, %2 : f64
// CHECK-NEXT:     %7 = divf %5, %6 : f64
// CHECK-NEXT:     %8 = mulf %cst, %4 : f64
// CHECK-NEXT:     %9 = divf %8, %6 : f64
// CHECK-NEXT:     %10 = negf %7 : f64
// CHECK-NEXT:     %11 = divf %10, %cst_0 : f64
// CHECK-NEXT:     %12 = addf %cst, %7 : f64
// CHECK-NEXT:     %13 = negf %9 : f64
// CHECK-NEXT:     %14 = divf %13, %cst_0 : f64
// CHECK-NEXT:     %15 = addf %cst, %9 : f64
// CHECK-NEXT:     %16 = index_cast %arg0 : i32 to index
// CHECK-NEXT:     %17 = negf %11 : f64
// CHECK-NEXT:     %18 = negf %14 : f64
// CHECK-NEXT:     %19 = mulf %cst_0, %14 : f64
// CHECK-NEXT:     %20 = addf %cst, %19 : f64
// CHECK-NEXT:     %21 = mulf %cst_0, %11 : f64
// CHECK-NEXT:     %22 = addf %cst, %21 : f64
// CHECK-NEXT:     affine.for %arg6 = 1 to #map0()[%16] {
// CHECK-NEXT:       affine.for %arg7 = 1 to #map1()[%0] {
// CHECK-NEXT:         affine.store %cst, %arg3[0, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:         affine.store %cst_1, %arg4[%arg7, 0] : memref<1000x1000xf64>
// CHECK-NEXT:         %23 = affine.load %arg3[0, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:         affine.store %23, %arg5[%arg7, 0] : memref<1000x1000xf64>
// CHECK-NEXT:         affine.for %arg8 = 1 to #map1()[%0] {
// CHECK-NEXT:           %24 = affine.load %arg4[%arg7, %arg8 - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %25 = mulf %11, %24 : f64
// CHECK-NEXT:           %26 = addf %25, %12 : f64
// CHECK-NEXT:           %27 = divf %17, %26 : f64
// CHECK-NEXT:           affine.store %27, %arg4[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:           %28 = affine.load %arg2[%arg8, %arg7 - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %29 = mulf %18, %28 : f64
// CHECK-NEXT:           %30 = affine.load %arg2[%arg8, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:           %31 = mulf %20, %30 : f64
// CHECK-NEXT:           %32 = addf %29, %31 : f64
// CHECK-NEXT:           %33 = affine.load %arg2[%arg8, %arg7 + 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %34 = mulf %14, %33 : f64
// CHECK-NEXT:           %35 = subf %32, %34 : f64
// CHECK-NEXT:           %36 = affine.load %arg5[%arg7, %arg8 - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %37 = mulf %11, %36 : f64
// CHECK-NEXT:           %38 = subf %35, %37 : f64
// CHECK-NEXT:           %39 = affine.load %arg4[%arg7, %arg8 - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %40 = mulf %11, %39 : f64
// CHECK-NEXT:           %41 = addf %40, %12 : f64
// CHECK-NEXT:           %42 = divf %38, %41 : f64
// CHECK-NEXT:           affine.store %42, %arg5[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         affine.store %cst, %arg3[symbol(%0) - 1, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:         affine.for %arg8 = 1 to #map1()[%0] {
// CHECK-NEXT:           %24 = affine.load %arg4[%arg7, -%arg8 + symbol(%0) - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %25 = affine.load %arg3[-%arg8 + symbol(%0), %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:           %26 = mulf %24, %25 : f64
// CHECK-NEXT:           %27 = affine.load %arg5[%arg7, -%arg8 + symbol(%0) - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %28 = addf %26, %27 : f64
// CHECK-NEXT:           affine.store %28, %arg3[-%arg8 + symbol(%0) - 1, %arg7] : memref<1000x1000xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.for %arg7 = 1 to #map1()[%0] {
// CHECK-NEXT:         affine.store %cst, %arg2[%arg7, 0] : memref<1000x1000xf64>
// CHECK-NEXT:         affine.store %cst_1, %arg4[%arg7, 0] : memref<1000x1000xf64>
// CHECK-NEXT:         %23 = affine.load %arg2[%arg7, 0] : memref<1000x1000xf64>
// CHECK-NEXT:         affine.store %23, %arg5[%arg7, 0] : memref<1000x1000xf64>
// CHECK-NEXT:         affine.for %arg8 = 1 to #map1()[%0] {
// CHECK-NEXT:           %24 = affine.load %arg4[%arg7, %arg8 - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %25 = mulf %14, %24 : f64
// CHECK-NEXT:           %26 = addf %25, %15 : f64
// CHECK-NEXT:           %27 = divf %18, %26 : f64
// CHECK-NEXT:           affine.store %27, %arg4[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:           %28 = affine.load %arg3[%arg7 - 1, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:           %29 = mulf %17, %28 : f64
// CHECK-NEXT:           %30 = affine.load %arg3[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:           %31 = mulf %22, %30 : f64
// CHECK-NEXT:           %32 = addf %29, %31 : f64
// CHECK-NEXT:           %33 = affine.load %arg3[%arg7 + 1, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:           %34 = mulf %11, %33 : f64
// CHECK-NEXT:           %35 = subf %32, %34 : f64
// CHECK-NEXT:           %36 = affine.load %arg5[%arg7, %arg8 - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %37 = mulf %14, %36 : f64
// CHECK-NEXT:           %38 = subf %35, %37 : f64
// CHECK-NEXT:           %39 = affine.load %arg4[%arg7, %arg8 - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %40 = mulf %14, %39 : f64
// CHECK-NEXT:           %41 = addf %40, %15 : f64
// CHECK-NEXT:           %42 = divf %38, %41 : f64
// CHECK-NEXT:           affine.store %42, %arg5[%arg7, %arg8] : memref<1000x1000xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         affine.store %cst, %arg2[%arg7, symbol(%0) - 1] : memref<1000x1000xf64>
// CHECK-NEXT:         affine.for %arg8 = 1 to #map1()[%0] {
// CHECK-NEXT:           %24 = affine.load %arg4[%arg7, -%arg8 + symbol(%0) - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %25 = affine.load %arg2[%arg7, -%arg8 + symbol(%0)] : memref<1000x1000xf64>
// CHECK-NEXT:           %26 = mulf %24, %25 : f64
// CHECK-NEXT:           %27 = affine.load %arg5[%arg7, -%arg8 + symbol(%0) - 1] : memref<1000x1000xf64>
// CHECK-NEXT:           %28 = addf %26, %27 : f64
// CHECK-NEXT:           affine.store %28, %arg2[%arg7, -%arg8 + symbol(%0) - 1] : memref<1000x1000xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// EXEC: {{[0-9]\.[0-9]+}}
