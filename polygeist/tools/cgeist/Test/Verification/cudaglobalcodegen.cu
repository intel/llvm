// RUN: cgeist %s -O0 -w --cuda-gpu-arch=sm_60 -nocudalib -nocudainc %resourcedir --function=* -S | FileCheck %s

#include "Inputs/cuda.h"

__global__ void bar(int * a)
{
#ifdef __CUDA_ARCH__
	*a = 1;
#else
	*a = 2;
#endif
}

void baz(int * a){
    bar<<<dim3(1,1,1), dim3(1,1,1)>>>(a);
}

// CHECK:  func private @_Z18__device_stub__barPi(%arg0: memref<?xi32>)
// CHECK-NEXT:    call @_Z3barPi(%arg0) : (memref<?xi32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func.func private @_Z3barPi(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    affine.store %c1_i32, %arg0[0] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func @_Z3bazPi(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:         gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = {{.*}}, %arg8 = {{.*}}, %arg9 = {{.*}}) threads(%arg4, %arg5, %arg6) in (%arg10 = {{.*}}, %arg11 = {{.*}}, %arg12 = {{.*}}) {
// CHECK-NEXT:      call @_Z18__device_stub__barPi(%arg0) : (memref<?xi32>) -> ()
// CHECK-NEXT:      gpu.terminator
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
