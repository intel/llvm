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

// CHECK-LABEL:   func.func @kernel_correlation(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32 {llvm.noundef}, %[[VAL_1:.*]]: f64 {llvm.noundef}, %[[VAL_2:.*]]: f64 {llvm.noundef}, %[[VAL_3:.*]]: memref<?x28xf64> {llvm.noalias, llvm.noundef}, %[[VAL_4:.*]]: memref<?x28xf64> {llvm.noalias, llvm.noundef}, %[[VAL_5:.*]]: memref<?xf64> {llvm.noalias, llvm.noundef}, %[[VAL_6:.*]]: memref<?xf64> {llvm.noalias, llvm.noundef}, %[[VAL_7:.*]]: memref<?xf64> {llvm.noalias, llvm.noundef})
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           affine.for %[[VAL_10:.*]] = 0 to %[[VAL_9]] {
// CHECK:             affine.store %[[VAL_8]], %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             affine.store %[[VAL_8]], %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             %[[VAL_11:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             %[[VAL_12:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             %[[VAL_13:.*]]:2 = affine.for %[[VAL_14:.*]] = 0 to %[[VAL_9]] iter_args(%[[VAL_15:.*]] = %[[VAL_11]], %[[VAL_16:.*]] = %[[VAL_12]]) -> (f64, f64) {
// CHECK:               %[[VAL_17:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_10]], %[[VAL_14]]] : memref<?x28xf64>
// CHECK:               %[[VAL_18:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_14]]] : memref<?xf64>
// CHECK:               %[[VAL_19:.*]] = math.fma %[[VAL_17]], %[[VAL_18]], %[[VAL_15]] : f64
// CHECK:               %[[VAL_20:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_10]], %[[VAL_14]]] : memref<?x28xf64>
// CHECK:               %[[VAL_21:.*]] = math.fma %[[VAL_20]], %[[VAL_18]], %[[VAL_16]] : f64
// CHECK:               affine.yield %[[VAL_19]], %[[VAL_21]] : f64, f64
// CHECK:             }
// CHECK:             affine.store %[[VAL_22:.*]]#1, %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             affine.store %[[VAL_22]]#0, %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             %[[VAL_23:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             %[[VAL_24:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:             %[[VAL_25:.*]] = arith.mulf %[[VAL_2]], %[[VAL_24]] : f64
// CHECK:             %[[VAL_26:.*]] = math.fma %[[VAL_1]], %[[VAL_23]], %[[VAL_25]] : f64
// CHECK:             affine.store %[[VAL_26]], %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<?xf64>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// FULLRANK: func @kernel_correlation(%{{.*}}: i32, %{{.*}}: f64, %{{.*}}: f64, %{{.*}}: memref<28x28xf64>, %{{.*}}: memref<28x28xf64>, %{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>)
