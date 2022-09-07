// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: cgeist %s %polyverify %stdinclude -O3 -o %s.execm -lm && %s.execm &> %s.out2
// RUN: rm -f %s.exec1 %s.execm
// RUN: diff %s.out1 %s.out2
// RUN: rm -f %s.out1 %s.out2
// RUN: cgeist %s %polyexec %stdinclude -O3 -o %s.execm -lm && %s.execm > %s.mlir.time; cat %s.mlir.time | FileCheck %s --check-prefix EXEC
// RUN: clang %s -O3 %polyexec %stdinclude -o %s.exec2 && %s.exec2 > %s.clang.time; cat %s.clang.time | FileCheck %s --check-prefix EXEC
// RUN: rm -f %s.exec2 %s.execm %s.mlir.time %s.clang.time

// RUN: clang %s -O3 %stdinclude %polyverify -o %s.exec1 && %s.exec1 &> %s.out1
// RUN: cgeist %s %polyverify %stdinclude -detect-reduction -O3 -o %s.execm -lm && %s.execm &> %s.out2
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
/* deriche.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "deriche.h"


/* Array initialization. */
static
void init_array (int w, int h, DATA_TYPE* alpha,
		 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
{
  int i, j;

  *alpha=0.25; //parameter of the filter

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int w, int h,
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
    }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static
void kernel_deriche(int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
    int i,j;
    DATA_TYPE xm1, tm1, ym1, ym2;
    DATA_TYPE xp1, xp2;
    DATA_TYPE tp1, tp2;
    DATA_TYPE yp1, yp2;

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

#pragma scop
   k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
   a1 = a5 = k;
   a2 = a6 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
   a3 = a7 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
   a4 = a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
   b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   c1 = c2 = 1;

   for (i=0; i<_PB_W; i++) {
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        xm1 = SCALAR_VAL(0.0);
        for (j=0; j<_PB_H; j++) {
            y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int w = W;
  int h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);


  /* Initialize array(s). */
  init_array (w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_deriche (w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(imgIn);
  POLYBENCH_FREE_ARRAY(imgOut);
  POLYBENCH_FREE_ARRAY(y1);
  POLYBENCH_FREE_ARRAY(y2);

  return 0;
}

// CHECK:   func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<4096x2160xf32>, %arg4: memref<4096x2160xf32>, %arg5: memref<4096x2160xf32>, %arg6: memref<4096x2160xf32>) {
// CHECK-NEXT:      %c0 = constant 0 : index
// CHECK-NEXT:      %cst = constant 1.000000e+00 : f32
// CHECK-NEXT:      %cst_0 = constant 2.000000e+00 : f32
// CHECK-NEXT:      %c1_i32 = constant 1 : i32
// CHECK-NEXT:      %cst_1 = constant 0.000000e+00 : f32
// CHECK-NEXT:      %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:      %1 = index_cast %arg0 : i32 to index
// CHECK-NEXT:      %2 = alloca() : memref<1xf32>
// CHECK-NEXT:      %3 = alloca() : memref<1xf32>
// CHECK-NEXT:      %4 = alloca() : memref<1xf32>
// CHECK-NEXT:      %5 = alloca() : memref<1xf32>
// CHECK-NEXT:      %6 = alloca() : memref<1xf32>
// CHECK-NEXT:      %7 = alloca() : memref<1xf32>
// CHECK-NEXT:      %8 = alloca() : memref<1xf32>
// CHECK-NEXT:      %9 = alloca() : memref<1xf32>
// CHECK-NEXT:      %10 = alloca() : memref<1xf32>
// CHECK-NEXT:      %11 = alloca() : memref<1xf32>
// CHECK-NEXT:      %12 = negf %arg2 : f32
// CHECK-NEXT:      %13 = exp %12 : f32
// CHECK-NEXT:      %14 = subf %cst, %13 : f32
// CHECK-NEXT:      %15 = mulf %14, %14 : f32
// CHECK-NEXT:      %16 = mulf %cst_0, %arg2 : f32
// CHECK-NEXT:      %17 = mulf %16, %13 : f32
// CHECK-NEXT:      %18 = addf %cst, %17 : f32
// CHECK-NEXT:      %19 = exp %16 : f32
// CHECK-NEXT:      %20 = subf %18, %19 : f32
// CHECK-NEXT:      %21 = divf %15, %20 : f32
// CHECK-NEXT:      %22 = mulf %21, %13 : f32
// CHECK-NEXT:      %23 = subf %arg2, %cst : f32
// CHECK-NEXT:      %24 = mulf %22, %23 : f32
// CHECK-NEXT:      %25 = addf %arg2, %cst : f32
// CHECK-NEXT:      %26 = mulf %22, %25 : f32
// CHECK-NEXT:      %27 = negf %21 : f32
// CHECK-NEXT:      %28 = negf %cst_0 : f32
// CHECK-NEXT:      %29 = mulf %28, %arg2 : f32
// CHECK-NEXT:      %30 = exp %29 : f32
// CHECK-NEXT:      %31 = mulf %27, %30 : f32
// CHECK-NEXT:      %32 = llvm.mlir.cast %cst_0 : f32 to !llvm.float
// CHECK-NEXT:      %33 = llvm.mlir.cast %12 : f32 to !llvm.float
// CHECK-NEXT:      %34 = "llvm.intr.pow"(%32, %33) : (!llvm.float, !llvm.float) -> !llvm.float
// CHECK-NEXT:      %35 = llvm.mlir.cast %34 : !llvm.float to f32
// CHECK-NEXT:      %36 = negf %30 : f32
// CHECK-NEXT:      %37 = sitofp %c1_i32 : i32 to f32
// CHECK-NEXT:      store %cst_1, %4[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %5[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %2[%c0] : memref<1xf32>
// CHECK-NEXT:      %38 = load %2[%c0] : memref<1xf32>
// CHECK-NEXT:      %39 = mulf %24, %38 : f32
// CHECK-NEXT:      %40 = load %4[%c0] : memref<1xf32>
// CHECK-NEXT:      %41 = mulf %35, %40 : f32
// CHECK-NEXT:      %42 = load %5[%c0] : memref<1xf32>
// CHECK-NEXT:      %43 = mulf %36, %42 : f32
// CHECK-NEXT:      %44 = load %4[%c0] : memref<1xf32>
// CHECK-NEXT:      store %44, %5[%c0] : memref<1xf32>
// CHECK-NEXT:      affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:        affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:          %78 = affine.load %arg3[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:          %79 = mulf %21, %78 : f32
// CHECK-NEXT:          %80 = addf %79, %39 : f32
// CHECK-NEXT:          %81 = addf %80, %41 : f32
// CHECK-NEXT:          %82 = addf %81, %43 : f32
// CHECK-NEXT:          affine.store %82, %arg5[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:          %83 = affine.load %arg3[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:          affine.store %83, %2[0] : memref<1xf32>
// CHECK-NEXT:          %84 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:          affine.store %84, %4[0] : memref<1xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      store %cst_1, %10[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %11[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %6[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %7[%c0] : memref<1xf32>
// CHECK-NEXT:      %45 = load %6[%c0] : memref<1xf32>
// CHECK-NEXT:      %46 = mulf %26, %45 : f32
// CHECK-NEXT:      %47 = load %7[%c0] : memref<1xf32>
// CHECK-NEXT:      %48 = mulf %31, %47 : f32
// CHECK-NEXT:      %49 = addf %46, %48 : f32
// CHECK-NEXT:      %50 = load %10[%c0] : memref<1xf32>
// CHECK-NEXT:      %51 = mulf %35, %50 : f32
// CHECK-NEXT:      %52 = addf %49, %51 : f32
// CHECK-NEXT:      %53 = load %11[%c0] : memref<1xf32>
// CHECK-NEXT:      %54 = mulf %36, %53 : f32
// CHECK-NEXT:      %55 = addf %52, %54 : f32
// CHECK-NEXT:      %56 = load %6[%c0] : memref<1xf32>
// CHECK-NEXT:      store %56, %7[%c0] : memref<1xf32>
// CHECK-NEXT:      %57 = load %10[%c0] : memref<1xf32>
// CHECK-NEXT:      store %57, %11[%c0] : memref<1xf32>
// CHECK-NEXT:      affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:        affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:          affine.store %55, %arg6[%arg7, -%arg8 + symbol(%0) - 1] : memref<4096x2160xf32>
// CHECK-NEXT:          %78 = affine.load %arg3[%arg7, -%arg8 + symbol(%0) - 1] : memref<4096x2160xf32>
// CHECK-NEXT:          affine.store %78, %6[0] : memref<1xf32>
// CHECK-NEXT:          %79 = affine.load %arg6[%arg7, -%arg8 + symbol(%0) - 1] : memref<4096x2160xf32>
// CHECK-NEXT:          affine.store %79, %10[0] : memref<1xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:        affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:          %78 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:          %79 = affine.load %arg6[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:          %80 = addf %78, %79 : f32
// CHECK-NEXT:          %81 = mulf %37, %80 : f32
// CHECK-NEXT:          affine.store %81, %arg4[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      store %cst_1, %3[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %4[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %5[%c0] : memref<1xf32>
// CHECK-NEXT:      %58 = load %3[%c0] : memref<1xf32>
// CHECK-NEXT:      %59 = mulf %24, %58 : f32
// CHECK-NEXT:      %60 = load %4[%c0] : memref<1xf32>
// CHECK-NEXT:      %61 = mulf %35, %60 : f32
// CHECK-NEXT:      %62 = load %5[%c0] : memref<1xf32>
// CHECK-NEXT:      %63 = mulf %36, %62 : f32
// CHECK-NEXT:      %64 = load %4[%c0] : memref<1xf32>
// CHECK-NEXT:      store %64, %5[%c0] : memref<1xf32>
// CHECK-NEXT:      affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:        affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:          %78 = affine.load %arg4[%arg8, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:          %79 = mulf %21, %78 : f32
// CHECK-NEXT:          %80 = addf %79, %59 : f32
// CHECK-NEXT:          %81 = addf %80, %61 : f32
// CHECK-NEXT:          %82 = addf %81, %63 : f32
// CHECK-NEXT:          affine.store %82, %arg5[%arg8, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:          %83 = affine.load %arg4[%arg8, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:          affine.store %83, %3[0] : memref<1xf32>
// CHECK-NEXT:          %84 = affine.load %arg5[%arg8, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:          affine.store %84, %4[0] : memref<1xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      store %cst_1, %8[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %9[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %10[%c0] : memref<1xf32>
// CHECK-NEXT:      store %cst_1, %11[%c0] : memref<1xf32>
// CHECK-NEXT:      %65 = load %8[%c0] : memref<1xf32>
// CHECK-NEXT:      %66 = mulf %26, %65 : f32
// CHECK-NEXT:      %67 = load %9[%c0] : memref<1xf32>
// CHECK-NEXT:      %68 = mulf %31, %67 : f32
// CHECK-NEXT:      %69 = addf %66, %68 : f32
// CHECK-NEXT:      %70 = load %10[%c0] : memref<1xf32>
// CHECK-NEXT:      %71 = mulf %35, %70 : f32
// CHECK-NEXT:      %72 = addf %69, %71 : f32
// CHECK-NEXT:      %73 = load %11[%c0] : memref<1xf32>
// CHECK-NEXT:      %74 = mulf %36, %73 : f32
// CHECK-NEXT:      %75 = addf %72, %74 : f32
// CHECK-NEXT:      %76 = load %8[%c0] : memref<1xf32>
// CHECK-NEXT:      store %76, %9[%c0] : memref<1xf32>
// CHECK-NEXT:      %77 = load %10[%c0] : memref<1xf32>
// CHECK-NEXT:      store %77, %11[%c0] : memref<1xf32>
// CHECK-NEXT:      affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:        affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:          affine.store %75, %arg6[-%arg8 + symbol(%1) - 1, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:          %78 = affine.load %arg4[-%arg8 + symbol(%1) - 1, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:          affine.store %78, %8[0] : memref<1xf32>
// CHECK-NEXT:          %79 = affine.load %arg6[-%arg8 + symbol(%1) - 1, %arg7] : memref<4096x2160xf32>
// CHECK-NEXT:          affine.store %79, %10[0] : memref<1xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:        affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:          %78 = affine.load %arg5[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:          %79 = affine.load %arg6[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:          %80 = addf %78, %79 : f32
// CHECK-NEXT:          %81 = mulf %37, %80 : f32
// CHECK-NEXT:          affine.store %81, %arg4[%arg7, %arg8] : memref<4096x2160xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// EXEC: {{[0-9]\.[0-9]+}}
