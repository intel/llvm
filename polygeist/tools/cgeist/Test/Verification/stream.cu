// RUN: cgeist %s -O0 -w --cuda-gpu-arch=sm_60 -nocudalib -nocudainc %resourcedir --function=* -S | FileCheck %s

#include "Inputs/cuda.h"

__device__ void something(int* array, int n);

// Type your code here, or load an example.
__global__ void square(int *array, int n) {
	something(array, n);
}

void run(cudaStream_t stream1, int *array, int n) {
    square<<< 10, 20, 0, stream1>>> (array, n) ;
}

// CHECK:   func.func @_Z3runP10cudaStreamPii(%arg0: !llvm.ptr, %arg1: memref<?xi32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:          %20 = "polygeist.stream2token"(%arg0) : (!llvm.ptr) -> !gpu.async.token
// CHECK-NEXT:     %21 = gpu.launch async [%20] blocks(%arg3, %arg4, %arg5) in (%arg9 = {{.*}}, %arg10 = {{.*}}, %arg11 = {{.*}}) threads(%arg6, %arg7, %arg8) in (%arg12 = {{.*}}, %arg13 = {{.*}}, %arg14 = {{.*}}) {
// CHECK-NEXT:       func.call @_Z21__device_stub__squarePii(%arg1, %arg2) : (memref<?xi32>, i32) -> ()
// CHECK-NEXT:       gpu.terminator
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
