// RUN: cgeist -O2 %s --function=* -S -enable-attributes | FileCheck %s

void sum(double * __restrict__ result, double * __restrict__ array) {
    result[0] = 0;
    #pragma scop
    for (int i=0; i<10; i++) {
        result[0] += array[i];
    }
    #pragma endscop
}

// CHECK:  func @sum(%arg0: memref<?xf64>{{.*}}llvm.noalias{{.*}}, %arg1: memref<?xf64>{{.*}}llvm.noalias{{.*}})
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    affine.store %cst, %arg0[0] : memref<?xf64>
// CHECK-NEXT:    %[[i1:.+]] = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:    %[[i2:.+]] = affine.for %arg2 = 0 to 10 iter_args(%arg3 = %[[i1]]) -> (f64) {
// CHECK-NEXT:      %[[i3:.+]] = affine.load %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:      %[[i4:.+]] = arith.addf %arg3, %[[i3]] : f64
// CHECK-NEXT:      affine.yield %[[i4]] : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.store %[[i2]], %arg0[0] : memref<?xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
