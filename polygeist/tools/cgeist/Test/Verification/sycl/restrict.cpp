// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR-DAG: func.func @test_int(%arg0: memref<?xi32, 4> {{{.*}}llvm.noalias{{.*}}}, %arg1: memref<?xi32, 4> {{{.*}}llvm.noalias{{.*}}}) 
// CHECK-LLVM-DAG: define spir_func {{.*}}i32 @test_int(ptr addrspace(4) noalias {{.*}}%0, ptr addrspace(4) noalias {{.*}}%1)

extern "C" SYCL_EXTERNAL int test_int(int * __restrict__ a, int * __restrict__ b) {}

// CHECK-MLIR-DAG: func.func @test_struct(%arg0: !llvm.ptr<4> {{{.*}}llvm.noalias{{.*}}}, %arg1: !llvm.ptr<4> {{{.*}}llvm.noalias{{.*}}})
// CHECK-LLVM-DAG: define spir_func void @test_struct(ptr addrspace(4) noalias {{.*}}%0, ptr addrspace(4) noalias {{.*}}%1)
struct S {
  int i;
};
extern "C" SYCL_EXTERNAL void test_struct(struct S * __restrict__ a, struct S * __restrict__ b) {}

// CHECK-MLIR-DAG: func.func @test_vec(%arg0: memref<?x!sycl_vec_f64_16_, 4> {{{.*}}llvm.noalias{{.*}}}, %arg1: memref<?x!sycl_vec_f64_16_, 4> {{{.*}}llvm.noalias{{.*}}})
// CHECK-LLVM-DAG: define spir_func void @test_vec(ptr addrspace(4) noalias {{.*}}%0, ptr addrspace(4) noalias {{.*}}%1)
extern "C" SYCL_EXTERNAL void test_vec(sycl::vec<sycl::cl_double, 16> * __restrict__ a, const sycl::vec<sycl::cl_double, 16> * __restrict__ b) {}

// CHECK-MLIR-DAG: func.func @_ZN1B10test_classEP1APS_(%arg0: !llvm.ptr<4> {{.*}}, %arg1: !llvm.ptr<4> {llvm.noalias{{.*}}}, %arg2: !llvm.ptr<4> {llvm.noalias{{.*}}})
// CHECK-LLVM-DAG: define linkonce_odr spir_func void @_ZN1B10test_classEP1APS_(ptr addrspace(4) {{.*}}%0, ptr addrspace(4) noalias {{.*}}%1, ptr addrspace(4) noalias {{.*}}%2)
class A {};
class B {
  SYCL_EXTERNAL void test_class(class A * __restrict__ a, class B * __restrict__ b) {}
};

// CHECK-MLIR-DAG: gpu.func @{{.*}}kernel_args_restrict(%arg0: memref<?xi32, 1> {{{.*}}llvm.noalias{{.*}}}, %arg1: memref<?x!sycl_range_1_>{{.*}}, %arg2: memref<?x!sycl_range_1_>{{.*}}, %arg3: memref<?x!sycl_id_1_>{{.*}}, %arg4: memref<?xi32, 1> {{{.*}}llvm.noalias{{.*}}}, %arg5: memref<?x!sycl_range_1_>{{.*}}, %arg6: memref<?x!sycl_range_1_>{{.*}}, %arg7: memref<?x!sycl_id_1_>{{.*}})
// CHECK-LLVM-DAG: define weak_odr spir_kernel void @{{.*}}kernel_args_restrict(ptr addrspace(1) {{.*}}noalias{{.*}} %0, ptr {{.*}} %1, ptr {{.*}} %2, ptr {{.*}} %3, ptr addrspace(1) {{.*}}noalias{{.*}} %4, ptr {{.*}} %5, ptr {{.*}} %6, ptr {{.*}} %7)
using namespace sycl;
int args_restrict(std::array<int, 1> &A, std::array<int, 1> &B) {
  queue q;
  {
    auto bufA = buffer<int, 1>{A.data(), 1};
    auto bufB = buffer<int, 1>{B.data(), 1};
    q.submit([&](handler &cgh) {
      auto A = bufA.get_access<access::mode::write>(cgh);
      auto B = bufB.get_access<access::mode::read>(cgh);
      cgh.single_task<class kernel_args_restrict>(
          [=]() [[intel::kernel_args_restrict]]{
            A[0] = B[0];
          });

    });
  }
  return 0;
}
