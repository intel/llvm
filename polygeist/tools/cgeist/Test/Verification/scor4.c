// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s
// RUN: cgeist %s --function=kernel_correlation --raise-scf-to-affine -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

void use(int i);

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(double A[28], double B[28])
{
  int i;
  for (i = 1; i < 10; i++) {
    A[i] = 0.;
  }
  for (i = 1; i < 10; i++) {
    B[i] = 0.;
  }
}

// CHECK:   func @kernel_correlation(%arg0: memref<?xf64>, %arg1: memref<?xf64>)
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:       affine.for %arg2 = 1 to 10 {
// CHECK-NEXT:         affine.store %cst, %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.for %arg2 = 1 to 10 {
// CHECK-NEXT:         affine.store %cst, %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// FULLRANK:   func @kernel_correlation(%{{.*}}: memref<28xf64>, %{{.*}}: memref<28xf64>)
