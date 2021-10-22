// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

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
// CHECK: define dso_local spir_kernel void @{{.*}}basic(%struct.__generated_B* noundef byval(%struct.__generated_B) align 8 %_arg_Obj)
//
// Kernel object clone.
// CHECK: %[[K:[a-zA-Z0-9_.]+]] = alloca %union.__wrapper_union
// CHECK: %[[K_as_cast:[a-zA-Z0-9_.]+]] = addrspacecast %union.__wrapper_union* %[[K]] to %union.__wrapper_union addrspace(4)*
//
// Argument reference.
// CHECK: %[[Arg_ref:[a-zA-Z0-9_.]+]] = addrspacecast %struct.__generated_B* %_arg_Obj to %struct.__generated_B addrspace(4)*

// Initialization.
// CHECK: %[[K_BC:[a-zA-Z0-9_.]+]] = bitcast %union.__wrapper_union addrspace(4)* %[[K_as_cast]] to %class.anon addrspace(4)*
// CHECK: %[[GEP:[a-zA-Z0-9_.]+]] = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %[[K_BC]], i32 0, i32 0
// CHECK: %[[GEPBC:[a-zA-Z0-9_.]+]] = bitcast %struct.B addrspace(4)* %[[GEP]] to i8 addrspace(4)*
// CHECK: %[[ArgBC:[a-zA-Z0-9_.]+]] = bitcast %struct.__generated_B addrspace(4)* %[[Arg_ref]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %[[GEPBC]], i8 addrspace(4)* align 8 %[[ArgBC]], i64 16, i1 false)
//
// Kernel body call.
// CHECK: %[[K_BC:[a-zA-Z0-9_.]+]] = bitcast %union.__wrapper_union addrspace(4)* %[[K_as_cast]] to %class.anon addrspace(4)*
// CHECK: store %class.anon addrspace(4)* %[[K_BC]], %class.anon addrspace(4)* addrspace(4)* %[[K_STACK_PTR:[a-zA-Z0-9_.]+]]
// CHECK: %[[K_PTR:[a-zA-Z0-9_.]+]] = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* %[[K_STACK_PTR]]
// CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(16) %[[K_PTR]])

// CHECK: define dso_local spir_kernel void @{{.*}}nns(%struct.__generated_B.0* noundef byval(%struct.__generated_B.0) align 8 %_arg_NNSObj)
//
// Kernel object clone.
// CHECK: %[[NNSK:[a-zA-Z0-9_.]+]] = alloca %union.__wrapper_union.2
// CHECK: %[[NNSK_as_cast:[a-zA-Z0-9_.]+]] = addrspacecast %union.__wrapper_union.2* %[[NNSK]] to %union.__wrapper_union.2 addrspace(4)*
//
// Argument reference.
// CHECK: %[[NNSArg_ref:[a-zA-Z0-9_.]+]] = addrspacecast %struct.__generated_B.0* %_arg_NNSObj to %struct.__generated_B.0 addrspace(4)*
//
// Initialization.
// CHECK: %[[NNSK_BC:[a-zA-Z0-9_.]+]] = bitcast %union.__wrapper_union.2 addrspace(4)* %[[NNSK_as_cast]] to %class.anon.3 addrspace(4)*
// CHECK: %[[NNSGEP:[a-zA-Z0-9_.]+]] = getelementptr inbounds %class.anon.3, %class.anon.3 addrspace(4)* %[[NNSK_BC]], i32 0, i32 0
// CHECK: %[[NNSGEPBC:[a-zA-Z0-9_.]+]] = bitcast %struct.B addrspace(4)* %[[NNSGEP]] to i8 addrspace(4)*
// CHECK: %[[NNSArgBC:[a-zA-Z0-9_.]+]] = bitcast %struct.__generated_B.0 addrspace(4)* %[[NNSArg_ref]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %[[NNSGEPBC]], i8 addrspace(4)* align 8 %[[NNSArgBC]], i64 16, i1 false)
//
// Kernel body call.
// CHECK: %[[NNSK_BC:[a-zA-Z0-9_.]+]] = bitcast %union.__wrapper_union.2 addrspace(4)* %[[NNSK_as_cast]] to %class.anon.3 addrspace(4)*
// CHECK: store %class.anon.3 addrspace(4)* %[[NNSK_BC]], %class.anon.3 addrspace(4)* addrspace(4)* %[[NNSK_STACK_PTR:[a-zA-Z0-9_.]+]]
// CHECK: %[[NNSK_PTR:[a-zA-Z0-9_.]+]] = load %class.anon.3 addrspace(4)*, %class.anon.3 addrspace(4)* addrspace(4)* %[[NNSK_STACK_PTR]]
// CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_ENKUlvE_clEv(%class.anon.3 addrspace(4)* noundef align 8 dereferenceable_or_null(16) %[[NNSK_PTR]])
