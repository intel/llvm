// RUN: cgeist %s -O2 --function=kernel_correlation -S -enable-attributes | FileCheck %s
// RUN: cgeist %s -O2 --function=kernel_correlation -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int n, double alpha, double beta,
                    double A[restrict 28][28],
                    double B[restrict 28][28],
                    double tmp[restrict 28],
                    double x[restrict 28],
                    double y[restrict 28])
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      y[i] = SCALAR_VAL(0.0);
      for (j = 0; j < n; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }

}

// CHECK:   func @{{.*}}kernel_correlation{{.*}}(%arg0: i32{{.*}}, %arg1: f64{{.*}}, %arg2: f64{{.*}}, %arg3: memref<?x28xf64>{{.*}}llvm.noalias{{.*}}, %arg4: memref<?x28xf64>{{.*}}llvm.noalias{{.*}}, %arg5: memref<?xf64>{{.*}}llvm.noalias{{.*}}, %arg6: memref<?xf64>{{.*}}llvm.noalias{{.*}}, %arg7: memref<?xf64>{{.*}}llvm.noalias{{.*}})
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:       affine.store %cst, %arg5[%arg8] : memref<?xf64>
// CHECK-NEXT:       affine.store %cst, %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:       %1 = affine.load %arg5[%arg8] : memref<?xf64>
// CHECK-NEXT:       %2 = affine.load %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:       %3:2 = affine.for %arg9 = 0 to %0 iter_args(%arg10 = %1, %arg11 = %2) -> (f64, f64) {
// CHECK-NEXT:         %9 = affine.load %arg3[%arg8, %arg9] : memref<?x28xf64>
// CHECK-NEXT:         %10 = affine.load %arg6[%arg9] : memref<?xf64>
// CHECK-NEXT:         %11 = arith.mulf %9, %10 : f64
// CHECK-NEXT:         %12 = arith.addf %11, %arg10 : f64
// CHECK-NEXT:         %13 = affine.load %arg4[%arg8, %arg9] : memref<?x28xf64>
// CHECK-NEXT:         %14 = arith.mulf %13, %10 : f64
// CHECK-NEXT:         %15 = arith.addf %14, %arg11 : f64
// CHECK-NEXT:         affine.yield %12, %15 : f64, f64
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %3#1, %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:       affine.store %3#0, %arg5[%arg8] : memref<?xf64>
// CHECK-NEXT:       %4 = affine.load %arg5[%arg8] : memref<?xf64>
// CHECK-NEXT:       %5 = arith.mulf %arg1, %4 : f64
// CHECK-NEXT:       %6 = affine.load %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:       %7 = arith.mulf %arg2, %6 : f64
// CHECK-NEXT:       %8 = arith.addf %5, %7 : f64
// CHECK-NEXT:       affine.store %8, %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// FULLRANK: func @kernel_correlation(%{{.*}}: i32, %{{.*}}: f64, %{{.*}}: f64, %{{.*}}: memref<28x28xf64>, %{{.*}}: memref<28x28xf64>, %{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>)
