// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: func.func @test_int(%arg0: memref<?xi32, 4> {{{.*}}llvm.noalias{{.*}}}, %arg1: memref<?xi32, 4> {{{.*}}llvm.noalias{{.*}}}) 
// CHECK-LLVM: define spir_func {{.*}}i32 @test_int(i32 addrspace(4)* noalias {{.*}}%0, i32 addrspace(4)* noalias {{.*}}%1)

extern "C" SYCL_EXTERNAL int test_int(int * __restrict__ a, int * __restrict__ b) {}

// CHECK-MLIR: func.func @test_struct(%arg0: !llvm.ptr<struct<(i32)>, 4> {{{.*}}llvm.noalias{{.*}}}, %arg1: !llvm.ptr<struct<(i32)>, 4> {{{.*}}llvm.noalias{{.*}}})
// CHECK-LLVM: define spir_func void @test_struct({ i32 } addrspace(4)* noalias {{.*}}%0, { i32 } addrspace(4)* noalias {{.*}}%1)
struct S {
  int i;
};
extern "C" SYCL_EXTERNAL void test_struct(struct S * __restrict__ a, struct S * __restrict__ b) {}

// CHECK-MLIR: func.func @test_vec(%arg0: memref<?x!sycl_vec_f64_16_, 4> {{{.*}}llvm.noalias{{.*}}}, %arg1: memref<?x!sycl_vec_f64_16_, 4> {{{.*}}llvm.noalias{{.*}}})
// CHECK-LLVM: define spir_func void @test_vec(%"class.sycl::_V1::vec" addrspace(4)* noalias {{.*}}%0, %"class.sycl::_V1::vec" addrspace(4)* noalias {{.*}}%1)
extern "C" SYCL_EXTERNAL void test_vec(sycl::vec<sycl::cl_double, 16> * __restrict__ a, const sycl::vec<sycl::cl_double, 16> * __restrict__ b) {}
