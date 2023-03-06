// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-MLIR: func.func @func(%arg0: memref<?xi32, 4> {{{.*}}llvm.noalias{{.*}}}, %arg1: memref<?xi32, 4> {{{.*}}llvm.noalias{{.*}}}) 
// CHECK-LLVM: define spir_func {{.*}}i32 @func(i32 addrspace(4)* noalias {{.*}}%0, i32 addrspace(4)* noalias {{.*}}%1)

#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL int func(int * __restrict__ a, int * __restrict__ b) {
  return *a + *b;
}
