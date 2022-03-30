// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that compiler generates correct address spaces for pointer
// kernel arguments that are wrapped by struct.

#include "Inputs/sycl.hpp"

struct A {
  float *F;
};

struct B {
  int *F1;
  float *F2;
  A F3;
  int *F4[2];
};

int main() {
  B Obj;
  cl::sycl::kernel_single_task<class structs>(
      [=]() {
        (void)Obj;
      });
  float A = 1;
  float *Ptr = &A;
  auto Lambda = [=]() {
    *Ptr += 1;
  };
  cl::sycl::kernel_single_task<class lambdas>([=]() {
    Lambda();
  });
  return 0;
}

// CHECK: %[[WRAPPER_F1:[a-zA-Z0-9_.]+]] = type { i32 addrspace(1)* }
// CHECK: %[[WRAPPER_F2:[a-zA-Z0-9_.]+]] = type { float addrspace(1)* }
// CHECK: %[[WRAPPER_F:[a-zA-Z0-9_.]+]] = type { float addrspace(1)* }
// CHECK: %[[WRAPPER_F4_1:[a-zA-Z0-9_.]+]] = type { i32 addrspace(1)* }
// CHECK: %[[WRAPPER_F4_2:[a-zA-Z0-9_.]+]] = type { i32 addrspace(1)* }
// CHECK: %[[WRAPPER_LAMBDA_PTR:[a-zA-Z0-9_.]+]] = type { float addrspace(1)* }
// CHECK: define {{.*}}spir_kernel void @{{.*}}structs
// CHECK-SAME: %[[WRAPPER_F1]]* noundef byval(%[[WRAPPER_F1]]) align 8 %_arg_F1,
// CHECK-SAME: %[[WRAPPER_F2]]* noundef byval(%[[WRAPPER_F2]]) align 8 %_arg_F2,
// CHECK-SAME: %[[WRAPPER_F]]* noundef byval(%[[WRAPPER_F]]) align 8 %_arg_F,
// CHECK-SAME: %[[WRAPPER_F4_1]]* noundef byval(%[[WRAPPER_F4_1]]) align 8 %_arg_F4
// CHECK-SAME: %[[WRAPPER_F4_2]]* noundef byval(%[[WRAPPER_F4_2]]) align 8 %_arg_F41
// CHECK: define {{.*}}spir_kernel void @{{.*}}lambdas{{.*}}(%[[WRAPPER_LAMBDA_PTR]]* noundef byval(%[[WRAPPER_LAMBDA_PTR]]) align 8 %_arg_Ptr)
