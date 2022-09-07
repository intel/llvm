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
/* fdtd-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "fdtd-2d.h"


/* Array initialization. */
static
void init_array (int tmax,
		 int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
	ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
	hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
		    int nx,
		    int ny,
		    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;

#pragma scop

  for(t = 0; t < _PB_TMAX; t++)
    {
      for (j = 0; j < _PB_NY; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < _PB_NX; i++)
	for (j = 0; j < _PB_NY; j++)
	  ey[i][j] = ey[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < _PB_NX; i++)
	for (j = 1; j < _PB_NY; j++)
	  ex[i][j] = ex[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < _PB_NX - 1; i++)
	for (j = 0; j < _PB_NY - 1; j++)
	  hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
    }

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);

  /* Initialize array(s). */
  init_array (tmax, nx, ny,
	      POLYBENCH_ARRAY(ex),
	      POLYBENCH_ARRAY(ey),
	      POLYBENCH_ARRAY(hz),
	      POLYBENCH_ARRAY(_fict_));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_fdtd_2d (tmax, nx, ny,
		  POLYBENCH_ARRAY(ex),
		  POLYBENCH_ARRAY(ey),
		  POLYBENCH_ARRAY(hz),
		  POLYBENCH_ARRAY(_fict_));


  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
				    POLYBENCH_ARRAY(ey),
				    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  return 0;
}

// CHECK: #map = affine_map<()[s0] -> (s0 - 1)>

// CHECK:    func @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<1000x1200xf64>, %arg4: memref<1000x1200xf64>, %arg5: memref<1000x1200xf64>, %arg6: memref<500xf64>) {
// CHECK-NEXT:    %cst = constant 5.000000e-01 : f64
// CHECK-NEXT:    %cst_0 = constant 0.69999999999999996 : f64
// CHECK-NEXT:    %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:    %1 = index_cast %arg2 : i32 to index
// CHECK-NEXT:    %2 = index_cast %arg0 : i32 to index
// CHECK-NEXT:    affine.for %arg7 = 0 to %2 {
// CHECK-NEXT:      %3 = affine.load %arg6[%arg7] : memref<500xf64>
// CHECK-NEXT:      affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:        affine.store %3, %arg4[0, %arg8] : memref<1000x1200xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 1 to %0 {
// CHECK-NEXT:        affine.for %arg9 = 0 to %1 {
// CHECK-NEXT:          %4 = affine.load %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg5[%arg8 - 1, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = mulf %cst, %7 : f64
// CHECK-NEXT:          %9 = subf %4, %8 : f64
// CHECK-NEXT:          affine.store %9, %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:        affine.for %arg9 = 1 to %1 {
// CHECK-NEXT:          %4 = affine.load %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg5[%arg8, %arg9 - 1] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = mulf %cst, %7 : f64
// CHECK-NEXT:          %9 = subf %4, %8 : f64
// CHECK-NEXT:          affine.store %9, %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg8 = 0 to #map()[%0] {
// CHECK-NEXT:        affine.for %arg9 = 0 to #map()[%1] {
// CHECK-NEXT:          %4 = affine.load %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %5 = affine.load %arg3[%arg8, %arg9 + 1] : memref<1000x1200xf64>
// CHECK-NEXT:          %6 = affine.load %arg3[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %7 = subf %5, %6 : f64
// CHECK-NEXT:          %8 = affine.load %arg4[%arg8 + 1, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %9 = addf %7, %8 : f64
// CHECK-NEXT:          %10 = affine.load %arg4[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:          %11 = subf %9, %10 : f64
// CHECK-NEXT:          %12 = mulf %cst_0, %11 : f64
// CHECK-NEXT:          %13 = subf %4, %12 : f64
// CHECK-NEXT:          affine.store %13, %arg5[%arg8, %arg9] : memref<1000x1200xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


// EXEC: {{[0-9]\.[0-9]+}}
