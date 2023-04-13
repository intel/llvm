// RUN: cgeist -O2 %s --function=* -S -enable-attributes | FileCheck %s

extern int print(double);

void sum(double * __restrict__ result, double * __restrict__ array, int N) {
    #pragma scop
    for (int j=0; j<N; j++) {
        result[0] = 0;
        for (int i=0; i<10; i++) {
            result[0] += array[i];
        }
        print(result[0]);
    }
    #pragma endscop
}

// CHECK:  func.func @sum(%arg0: memref<?xf64>{{.*}}llvm.noalias{{.*}}, %arg1: memref<?xf64>{{.*}}llvm.noalias{{.*}}, %arg2: i32{{.*}})
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %0 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:     affine.for %arg3 = 0 to %0 {
// CHECK-NEXT:       affine.store %cst, %arg0[0] : memref<?xf64>
// CHECK-NEXT:       %[[i2:.+]] = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:       %[[i3:.+]] = affine.for %arg4 = 0 to 10 iter_args(%arg5 = %[[i2]]) -> (f64) {
// CHECK-NEXT:         %[[i6:.+]] = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:         %[[i7:.+]] = arith.addf %arg5, %[[i6]] : f64
// CHECK-NEXT:         affine.yield %[[i7]] : f64
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %[[i3]], %arg0[0] : memref<?xf64>
// CHECK-NEXT:       %[[i4:.+]] = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:       %{{.*}} = func.call @print(%[[i4]]) : (f64) -> i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
