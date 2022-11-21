// RUN: cgeist %s -O2 --function=kernel_deriche -S | FileCheck %s

void kernel_deriche(int w, int h, double alpha, double** y2) {
    int i,j;

#pragma scop
    for (i=0; i<w; i++) {
        for (j=h-1; j>=0; j--) {
            y2[i][j] = alpha;
        }
    }
#pragma endscop
}
// CHECK:  func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?xmemref<?xf64>>)
// CHECK-NEXT:    %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:    %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    affine.for %arg4 = 0 to %1 {
// CHECK-NEXT:      affine.for %arg5 = 0 to %0 {
// CHECK-NEXT:        %2 = affine.load %arg3[%arg4] : memref<?xmemref<?xf64>>
// CHECK-NEXT:        affine.store %arg2, %2[-%arg5 + symbol(%0) - 1] : memref<?xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
