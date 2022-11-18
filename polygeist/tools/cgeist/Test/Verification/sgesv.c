// RUN: cgeist %s -O2 --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s
// RUN: cgeist %s -O2 --function=kernel_correlation --raise-scf-to-affine -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int n, double alpha, double beta,
                    double A[28][28],
                    double B[28][28],
                    double tmp[28],
                    double x[28],
                    double y[28])
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

// CHECK:   func @kernel_correlation(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x28xf64>, %arg4: memref<?x28xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>)
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg8 = 0 to %0 {
// CHECK-NEXT:       affine.store %cst, %arg5[%arg8] : memref<?xf64>
// CHECK-NEXT:       affine.store %cst, %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:       affine.for %arg9 = 0 to %0 {
// CHECK-NEXT:         %6 = affine.load %arg3[%arg8, %arg9] : memref<?x28xf64>
// CHECK-NEXT:         %7 = affine.load %arg6[%arg9] : memref<?xf64>
// CHECK-NEXT:         %8 = arith.mulf %6, %7 : f64
// CHECK-NEXT:         %9 = affine.load %arg5[%arg8] : memref<?xf64>
// CHECK-NEXT:         %10 = arith.addf %8, %9 : f64
// CHECK-NEXT:         affine.store %10, %arg5[%arg8] : memref<?xf64>
// CHECK-NEXT:         %11 = affine.load %arg4[%arg8, %arg9] : memref<?x28xf64>
// CHECK-NEXT:         %12 = affine.load %arg6[%arg9] : memref<?xf64>
// CHECK-NEXT:         %13 = arith.mulf %11, %12 : f64
// CHECK-NEXT:         %14 = affine.load %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:         %15 = arith.addf %13, %14 : f64
// CHECK-NEXT:         affine.store %15, %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %1 = affine.load %arg5[%arg8] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.mulf %arg1, %1 : f64
// CHECK-NEXT:       %3 = affine.load %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:       %4 = arith.mulf %arg2, %3 : f64
// CHECK-NEXT:       %5 = arith.addf %2, %4 : f64
// CHECK-NEXT:       affine.store %5, %arg7[%arg8] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// FULLRANK: func @kernel_correlation(%{{.*}}: i32, %{{.*}}: f64, %{{.*}}: f64, %{{.*}}: memref<28x28xf64>, %{{.*}}: memref<28x28xf64>, %{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>)
