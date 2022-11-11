// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s
// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int m, double corr[28][28])
{
    int i, j, k;
    //i = 0;
    for (i = 0; i < 28; i++)
    {
      for (j = i+1; j < m; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
        }
    }
}

// CHECK:   func @kernel_correlation(%arg0: i32, %arg1: memref<?x28xf64>)
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg2 = 0 to 28 {
// CHECK-NEXT:       affine.for %arg3 = #map1(%arg2) to %0 {
// CHECK-NEXT:         affine.store %cst, %arg1[%arg2, %arg3] : memref<?x28xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// FULLRANK:   func @kernel_correlation(%{{.*}}: i32, %{{.*}}: memref<28x28xf64>)
