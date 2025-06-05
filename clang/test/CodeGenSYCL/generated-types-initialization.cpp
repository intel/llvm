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
// CHECK: define dso_local spir_kernel void @{{.*}}basic(ptr noundef byval(%class.anon) align 8 %_arg__sycl_functor)
//
// Kernel body call.
// CHECK: %[[Obj_as_cast:[a-zA-Z0-9_.]+]] = addrspacecast ptr %_arg__sycl_functor to ptr addrspace(4)
// CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %[[Obj_as_cast]])

// CHECK: define dso_local spir_kernel void @{{.*}}nns(ptr noundef byval(%class.anon.0) align 8 %_arg__sycl_functor)
//
// Kernel body call.
// CHECK: %[[NNSK_as_cast:[a-zA-Z0-9_.]+]] = addrspacecast ptr %_arg__sycl_functor to ptr addrspace(4)
// CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_ENKUlvE_clEv(ptr addrspace(4) noundef align 8 dereferenceable_or_null(16) %[[NNSK_as_cast]])
