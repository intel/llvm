// RUN: cgeist %s -O2 --cuda-gpu-arch=sm_60 -nocudalib -nocudainc %resourcedir --function=* -S | FileCheck %s

#include "Inputs/cuda.h"

__device__ void something(int* array, int n);

// Type your code here, or load an example.
__global__ void square(int *array, int n) {
	something(array, n);
}

void run(cudaStream_t stream1, int *array, int n) {
    square<<< 10, 20, 0, stream1>>> (array, n) ;
}

// CHECK:   func.func @_Z3runP10cudaStreamPii(%arg0: !llvm.ptr<struct<()>>, %arg1: memref<?xi32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c10 = arith.constant 10 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c20 = arith.constant 20 : index
// CHECK-NEXT:     %0 = "polygeist.stream2token"(%arg0) : (!llvm.ptr<struct<()>>) -> !gpu.async.token
// CHECK-NEXT:     %1 = gpu.launch async [%0] blocks(%arg3, %arg4, %arg5) in (%arg9 = %c10, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c20, %arg13 = %c1, %arg14 = %c1) {
// CHECK-NEXT:       func.call @_Z21__device_stub__squarePii(%arg1, %arg2) : (memref<?xi32>, i32) -> ()
// CHECK-NEXT:       gpu.terminator
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
