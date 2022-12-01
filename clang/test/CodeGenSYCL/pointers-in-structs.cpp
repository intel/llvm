// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -opaque-pointers -emit-llvm %s -o - | FileCheck %s

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
  sycl::kernel_single_task<class structs>(
      [=]() {
        (void)Obj;
      });
  float A = 1;
  float *Ptr = &A;
  auto Lambda = [=]() {
    *Ptr += 1;
  };
  sycl::kernel_single_task<class lambdas>([=]() {
    Lambda();
  });
  return 0;
}

// CHECK: %[[GENERATED_B:[a-zA-Z0-9_.]+]] = type { ptr addrspace(1), ptr addrspace(1), %[[GENERATED_A:[a-zA-Z0-9_.]+]], [2 x ptr addrspace(1)] }
// CHECK: [[GENERATED_A]] = type { ptr addrspace(1) }
// CHECK: %[[WRAPPER_LAMBDA_PTR:[a-zA-Z0-9_.]+]] = type { ptr addrspace(1) }

// CHECK: define {{.*}}spir_kernel void @{{.*}}structs
// CHECK-SAME: ptr noundef byval(%[[GENERATED_B]]) align 8 %_arg_Obj
// CHECK: define {{.*}}spir_kernel void @{{.*}}lambdas{{.*}}(ptr noundef byval(%[[WRAPPER_LAMBDA_PTR]]) align 8 %_arg_Lambda)
