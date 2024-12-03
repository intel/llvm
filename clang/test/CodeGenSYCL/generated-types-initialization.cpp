// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that compiler generates correct code when kernel arguments
// are structs that contain pointers but not decomposed.

#include "sycl.hpp"

struct A {
  float *F;
};

struct B {
  int *F1;
  A F3;
  B(int *I, A AA) : F1(I), F3(AA) {};
};

struct Nested {
  typedef B TDA;
};

int main() {
  sycl::queue q;
  B Obj{nullptr, {nullptr}};

  q.submit([&](sycl::handler &h) {
  h.single_task<class basic>(
      [=]() {
        (void)Obj;
      });
  });

  Nested::TDA NNSObj{nullptr, {nullptr}};
  q.submit([&](sycl::handler &h) {
    h.single_task<class nns>([=]() {
      (void)NNSObj;
    });
  });
  return 0;
}
// CHECK: define dso_local spir_kernel void @{{.*}}basic(ptr noundef byval(%struct.__generated_B) align 8 %_arg_Obj)
//
// Kernel object clone.
// CHECK: %[[K:[a-zA-Z0-9_.]+]] = alloca %class.anon
// CHECK: %[[K_as_cast:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[K]] to ptr addrspace(4)
//
// Argument reference.
// CHECK: %[[Arg_ref:[a-zA-Z0-9_.]+]] = addrspacecast ptr %_arg_Obj to ptr addrspace(4)
//
// Initialization.
// CHECK: %[[GEP:[a-zA-Z0-9_.]+]] = getelementptr inbounds nuw %class.anon, ptr addrspace(4) %[[K_as_cast]], i32 0, i32 0
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %[[GEP]], ptr addrspace(4) align 8 %[[Arg_ref]], i64 16, i1 false)
//
// Kernel body call.
// CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %[[K_as_cast]])

// CHECK: define dso_local spir_kernel void @{{.*}}nns(ptr noundef byval(%struct.__generated_B.0) align 8 %_arg_NNSObj)
//
// Kernel object clone.
// CHECK: %[[NNSK:[a-zA-Z0-9_.]+]] = alloca %class.anon.2
// CHECK: %[[NNSK_as_cast:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[NNSK]] to ptr addrspace(4)
//
// Argument reference.
// CHECK: %[[NNSArg_ref:[a-zA-Z0-9_.]+]] = addrspacecast ptr %_arg_NNSObj to ptr addrspace(4)
//
// Initialization.
// CHECK: %[[NNSGEP:[a-zA-Z0-9_.]+]] = getelementptr inbounds nuw %class.anon.2, ptr addrspace(4) %[[NNSK_as_cast]], i32 0, i32 0
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %[[NNSGEP]], ptr addrspace(4) align 8 %[[NNSArg_ref]], i64 16, i1 false)
//
// Kernel body call.
// CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_ENKUlvE_clEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %[[NNSK_as_cast]])
