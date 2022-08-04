// RUN: cgeist %s --function=okernel_2mm -S | FileCheck %s

void okernel_2mm(unsigned int ni,
                 double *tmp) {
  int i, j, k;

#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (i = 0; i < ni; i++)
    {
      tmp[i] = 0.0;
    }
#pragma endscop
}

// CHECK:  func @okernel_2mm(%arg0: i32, %arg1: memref<?xf64>)
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    affine.for %arg2 = 0 to %0 {
// CHECK-NEXT:      affine.store %cst, %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
