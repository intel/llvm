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
/* correlation.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "correlation.h"


/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  DATA_TYPE eps = SCALAR_VAL(0.1);


#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < _PB_M; j++)
    {
      stddev[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
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
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_correlation (m, n, float_n,
		      POLYBENCH_ARRAY(data),
		      POLYBENCH_ARRAY(corr),
		      POLYBENCH_ARRAY(mean),
		      POLYBENCH_ARRAY(stddev));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}

// CHECK: #map0 = affine_map<()[s0] -> (s0 - 1)>
// CHECK-NEXT: #map1 = affine_map<(d0) -> (d0 + 1)>
// CHECK-NEXT: module
// CHECK-NEXT:   llvm.mlir.global internal constant @str7("==END   DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str6("\0Aend   dump: %s\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str5("%0.2lf \00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str4("\0A\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str3("corr\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str2("begin dump: %s\00")
// CHECK-NEXT:   llvm.mlir.global internal constant @str1("==BEGIN DUMP_ARRAYS==\0A\00")
// CHECK-NEXT:   llvm.mlir.global external @stderr() : !llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>
// CHECK-NEXT:   llvm.func @fprintf(!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, ...) -> !llvm.i32
// CHECK-NEXT:   llvm.mlir.global internal constant @str0("\00")
// CHECK-NEXT:   llvm.func @strcmp(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:   func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1400_i32 = constant 1400 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %false = constant false
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = alloca() : memref<1xf64>
// CHECK-NEXT:     %1 = alloc() : memref<1400x1200xf64>
// CHECK-NEXT:     %2 = alloc() : memref<1200x1200xf64>
// CHECK-NEXT:     %3 = alloc() : memref<1200xf64>
// CHECK-NEXT:     %4 = alloc() : memref<1200xf64>
// CHECK-NEXT:     %5 = memref_cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:     call @init_array(%c1200_i32, %c1400_i32, %5, %1) : (i32, i32, memref<?xf64>, memref<1400x1200xf64>) -> ()
// CHECK-NEXT:     %6 = load %0[%c0] : memref<1xf64>
// CHECK-NEXT:     call @kernel_correlation(%c1200_i32, %c1400_i32, %6, %1, %2, %3, %4) : (i32, i32, f64, memref<1400x1200xf64>, memref<1200x1200xf64>, memref<1200xf64>, memref<1200xf64>) -> ()
// CHECK-NEXT:     %7 = cmpi "sgt", %arg0, %c42_i32 : i32
// CHECK-NEXT:     %8 = scf.if %7 -> (i1) {
// CHECK-NEXT:       %9 = llvm.load %arg1 : !llvm.ptr<ptr<i8>>
// CHECK-NEXT:       %10 = llvm.mlir.addressof @str0 : !llvm.ptr<array<1 x i8>>
// CHECK-NEXT:       %11 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:       %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<1 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %13 = llvm.call @strcmp(%9, %12) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:       %14 = llvm.mlir.cast %13 : !llvm.i32 to i32
// CHECK-NEXT:       %15 = trunci %14 : i32 to i1
// CHECK-NEXT:       %16 = xor %15, %true : i1
// CHECK-NEXT:       scf.yield %16 : i1
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %false : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.if %8 {
// CHECK-NEXT:       call @print_array(%c1200_i32, %2) : (i32, memref<1200x1200xf64>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @init_array(%arg0: i32, %arg1: i32, %arg2: memref<?xf64>, %arg3: memref<1400x1200xf64>) {
// CHECK-NEXT:     %c0 = constant 0 : index
// CHECK-NEXT:     %c1400_i32 = constant 1400 : i32
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1200_i32 = constant 1200 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = sitofp %c1400_i32 : i32 to f64
// CHECK-NEXT:     store %0, %arg2[%c0] : memref<?xf64>
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %2 = cmpi "slt", %1, %c1400_i32 : i32
// CHECK-NEXT:     %3 = index_cast %1 : i32 to index
// CHECK-NEXT:     cond_br %2, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%4: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %5 = cmpi "slt", %4, %c1200_i32 : i32
// CHECK-NEXT:     %6 = index_cast %4 : i32 to index
// CHECK-NEXT:     cond_br %5, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %7 = muli %1, %4 : i32
// CHECK-NEXT:     %8 = sitofp %7 : i32 to f64
// CHECK-NEXT:     %9 = sitofp %c1200_i32 : i32 to f64
// CHECK-NEXT:     %10 = divf %8, %9 : f64
// CHECK-NEXT:     %11 = sitofp %1 : i32 to f64
// CHECK-NEXT:     %12 = addf %10, %11 : f64
// CHECK-NEXT:     store %12, %arg3[%3, %6] : memref<1400x1200xf64>
// CHECK-NEXT:     %13 = addi %4, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%13 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %14 = addi %1, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%14 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @kernel_correlation(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1400x1200xf64>, %arg4: memref<1200x1200xf64>, %arg5: memref<1200xf64>, %arg6: memref<1200xf64>) {
// CHECK-NEXT:     %cst = constant 1.000000e-01 : f64
// CHECK-NEXT:     %c1 = constant 1 : index
// CHECK-NEXT:     %cst_0 = constant 0.000000e+00 : f64
// CHECK-NEXT:     %cst_1 = constant 1.000000e+00 : f64
// CHECK-NEXT:     %0 = index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:       affine.store %cst_0, %arg5[%arg7] : memref<1200xf64>
// CHECK-NEXT:       affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:         %5 = affine.load %arg3[%arg8, %arg7] : memref<1400x1200xf64>
// CHECK-NEXT:         %6 = affine.load %arg5[%arg7] : memref<1200xf64>
// CHECK-NEXT:         %7 = addf %6, %5 : f64
// CHECK-NEXT:         affine.store %7, %arg5[%arg7] : memref<1200xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %3 = affine.load %arg5[%arg7] : memref<1200xf64>
// CHECK-NEXT:       %4 = divf %3, %arg2 : f64
// CHECK-NEXT:       affine.store %4, %arg5[%arg7] : memref<1200xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg7 = 0 to %1 {
// CHECK-NEXT:       affine.store %cst_0, %arg6[%arg7] : memref<1200xf64>
// CHECK-NEXT:       affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:         %8 = affine.load %arg3[%arg8, %arg7] : memref<1400x1200xf64>
// CHECK-NEXT:         %9 = affine.load %arg5[%arg7] : memref<1200xf64>
// CHECK-NEXT:         %10 = subf %8, %9 : f64
// CHECK-NEXT:         %11 = mulf %10, %10 : f64
// CHECK-NEXT:         %12 = affine.load %arg6[%arg7] : memref<1200xf64>
// CHECK-NEXT:         %13 = addf %12, %11 : f64
// CHECK-NEXT:         affine.store %13, %arg6[%arg7] : memref<1200xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %3 = affine.load %arg6[%arg7] : memref<1200xf64>
// CHECK-NEXT:       %4 = divf %3, %arg2 : f64
// CHECK-NEXT:       affine.store %4, %arg6[%arg7] : memref<1200xf64>
// CHECK-NEXT:       %5 = sqrt %4 : f64
// CHECK-NEXT:       affine.store %5, %arg6[%arg7] : memref<1200xf64>
// CHECK-NEXT:       %6 = cmpf "ule", %5, %cst : f64
// CHECK-NEXT:       %7 = scf.if %6 -> (f64) {
// CHECK-NEXT:         scf.yield %cst_1 : f64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %8 = affine.load %arg6[%arg7] : memref<1200xf64>
// CHECK-NEXT:         scf.yield %8 : f64
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %7, %arg6[%arg7] : memref<1200xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg7 = 0 to %0 {
// CHECK-NEXT:       affine.for %arg8 = 0 to %1 {
// CHECK-NEXT:         %3 = affine.load %arg5[%arg8] : memref<1200xf64>
// CHECK-NEXT:         %4 = affine.load %arg3[%arg7, %arg8] : memref<1400x1200xf64>
// CHECK-NEXT:         %5 = subf %4, %3 : f64
// CHECK-NEXT:         affine.store %5, %arg3[%arg7, %arg8] : memref<1400x1200xf64>
// CHECK-NEXT:         %6 = sqrt %arg2 : f64
// CHECK-NEXT:         %7 = affine.load %arg6[%arg8] : memref<1200xf64>
// CHECK-NEXT:         %8 = mulf %6, %7 : f64
// CHECK-NEXT:         %9 = divf %5, %8 : f64
// CHECK-NEXT:         affine.store %9, %arg3[%arg7, %arg8] : memref<1400x1200xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = subi %1, %c1 : index
// CHECK-NEXT:     affine.for %arg7 = 0 to #map0()[%1] {
// CHECK-NEXT:       affine.store %cst_1, %arg4[%arg7, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:       affine.for %arg8 = #map1(%arg7) to %1 {
// CHECK-NEXT:         affine.store %cst_0, %arg4[%arg7, %arg8] : memref<1200x1200xf64>
// CHECK-NEXT:         affine.for %arg9 = 0 to %0 {
// CHECK-NEXT:           %4 = affine.load %arg3[%arg9, %arg7] : memref<1400x1200xf64>
// CHECK-NEXT:           %5 = affine.load %arg3[%arg9, %arg8] : memref<1400x1200xf64>
// CHECK-NEXT:           %6 = mulf %4, %5 : f64
// CHECK-NEXT:           %7 = affine.load %arg4[%arg7, %arg8] : memref<1200x1200xf64>
// CHECK-NEXT:           %8 = addf %7, %6 : f64
// CHECK-NEXT:           affine.store %8, %arg4[%arg7, %arg8] : memref<1200x1200xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:         %3 = affine.load %arg4[%arg7, %arg8] : memref<1200x1200xf64>
// CHECK-NEXT:         affine.store %3, %arg4[%arg8, %arg7] : memref<1200x1200xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     store %cst_1, %arg4[%2, %2] : memref<1200x1200xf64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @print_array(%arg0: i32, %arg1: memref<1200x1200xf64>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %2 = llvm.mlir.addressof @str1 : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %3 = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:     %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %5 = llvm.call @fprintf(%1, %4) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %6 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %8 = llvm.mlir.addressof @str2 : !llvm.ptr<array<15 x i8>>
// CHECK-NEXT:     %9 = llvm.getelementptr %8[%3, %3] : (!llvm.ptr<array<15 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %10 = llvm.mlir.addressof @str3 : !llvm.ptr<array<5 x i8>>
// CHECK-NEXT:     %11 = llvm.getelementptr %10[%3, %3] : (!llvm.ptr<array<5 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %12 = llvm.call @fprintf(%7, %9, %11) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:   ^bb1(%13: i32):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:     %14 = cmpi "slt", %13, %arg0 : i32
// CHECK-NEXT:     %15 = index_cast %13 : i32 to index
// CHECK-NEXT:     cond_br %14, ^bb3(%c0_i32 : i32), ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %16 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %17 = llvm.load %16 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %18 = llvm.mlir.addressof @str6 : !llvm.ptr<array<17 x i8>>
// CHECK-NEXT:     %19 = llvm.getelementptr %18[%3, %3] : (!llvm.ptr<array<17 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %20 = llvm.mlir.addressof @str3 : !llvm.ptr<array<5 x i8>>
// CHECK-NEXT:     %21 = llvm.getelementptr %20[%3, %3] : (!llvm.ptr<array<5 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %22 = llvm.call @fprintf(%17, %19, %21) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     %23 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %24 = llvm.load %23 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %25 = llvm.mlir.addressof @str7 : !llvm.ptr<array<23 x i8>>
// CHECK-NEXT:     %26 = llvm.getelementptr %25[%3, %3] : (!llvm.ptr<array<23 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %27 = llvm.call @fprintf(%24, %26) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     return
// CHECK-NEXT:   ^bb3(%28: i32):  // 2 preds: ^bb1, ^bb4
// CHECK-NEXT:     %29 = cmpi "slt", %28, %arg0 : i32
// CHECK-NEXT:     %30 = index_cast %28 : i32 to index
// CHECK-NEXT:     cond_br %29, ^bb4, ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %31 = muli %13, %arg0 : i32
// CHECK-NEXT:     %32 = addi %31, %28 : i32
// CHECK-NEXT:     %33 = remi_signed %32, %c20_i32 : i32
// CHECK-NEXT:     %34 = cmpi "eq", %33, %c0_i32 : i32
// CHECK-NEXT:     scf.if %34 {
// CHECK-NEXT:       %44 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %45 = llvm.load %44 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:       %46 = llvm.mlir.addressof @str4 : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:       %47 = llvm.getelementptr %46[%3, %3] : (!llvm.ptr<array<2 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:       %48 = llvm.call @fprintf(%45, %47) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>) -> !llvm.i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %35 = llvm.mlir.addressof @stderr : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %36 = llvm.load %35 : !llvm.ptr<ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>>
// CHECK-NEXT:     %37 = llvm.mlir.addressof @str5 : !llvm.ptr<array<8 x i8>>
// CHECK-NEXT:     %38 = llvm.getelementptr %37[%3, %3] : (!llvm.ptr<array<8 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %39 = load %arg1[%15, %30] : memref<1200x1200xf64>
// CHECK-NEXT:     %40 = llvm.mlir.cast %39 : f64 to !llvm.double
// CHECK-NEXT:     %41 = llvm.call @fprintf(%36, %38, %40) : (!llvm.ptr<struct<"struct._IO_FILE", (i32, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<i8>, ptr<struct<"struct._IO_marker", opaque>>, ptr<struct<"struct._IO_FILE">>, i32, i32, i64, i16, i8, array<1 x i8>, ptr<i8>, i64, ptr<struct<"struct._IO_codecvt", opaque>>, ptr<struct<"struct._IO_wide_data", opaque>>, ptr<struct<"struct._IO_FILE">>, ptr<i8>, i64, i32, array<20 x i8>)>>, !llvm.ptr<i8>, !llvm.double) -> !llvm.i32
// CHECK-NEXT:     %42 = addi %28, %c1_i32 : i32
// CHECK-NEXT:     br ^bb3(%42 : i32)
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %43 = addi %13, %c1_i32 : i32
// CHECK-NEXT:     br ^bb1(%43 : i32)
// CHECK-NEXT:   }
// CHECK-NEXT: }

// EXEC: {{[0-9]\.[0-9]+}}
