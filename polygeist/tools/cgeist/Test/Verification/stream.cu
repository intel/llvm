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
// CHECK:       %12 = "polygeist.stream2token"(%arg0) : (!llvm.ptr<struct<()>>) -> !gpu.async.token
// CHECK-NEXT:       %13 = gpu.launch async [%12] blocks(%arg3, %arg4, %arg5) in (%arg9 = %1, %arg10 = %3, %arg11 = %5) threads(%arg6, %arg7, %arg8) in (%arg12 = %7, %arg13 = %9, %arg14 = %11) {
// CHECK-NEXT:       func.call @_Z21__device_stub__squarePii(%arg1, %arg2) : (memref<?xi32>, i32) -> ()
// CHECK-NEXT:       gpu.terminator
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
