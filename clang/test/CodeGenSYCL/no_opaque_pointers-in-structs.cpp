// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

// This test checks that compiler generates correct address spaces for pointer
// kernel arguments that are wrapped by struct. Generated class should retain
// original padding and alignment.

#include "Inputs/sycl.hpp"

struct A {
  float *F;
};

struct alignas(16) B {
  int *F1;
  float *F2;
  A F3;
  int *F4[2];
};

struct testFieldAlignment {
  int *ptr;
  alignas(16) float arr[4];
  int data;
};

int main() {
  B Obj1;
  testFieldAlignment Obj2;
  sycl::kernel_single_task<class structs>(
      [=]() {
        (void)Obj1;
        (void)Obj2;
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

// Padding in generated class class should match the 'original' padding
// CHECK: %[[GENERATED_B:[a-zA-Z0-9_.]+]] = type { i32 addrspace(1)*, float addrspace(1)*, %[[GENERATED_A:[a-zA-Z0-9_.]+]], [2 x i32 addrspace(1)*], [8 x i8] } 
// CHECK: %[[GENERATED_A]] = type { float addrspace(1)* }
// CHECK: %[[GENERATED_TESTFIELDALIGNMENT:[a-zA-Z0-9_.]+]] = type { i32 addrspace(1)*, [8 x i8], [4 x float], i32, [12 x i8] } 
// CHECK: %struct.B = type { i32 addrspace(4)*, float addrspace(4)*, %struct.A, [2 x i32 addrspace(4)*], [8 x i8] }
// CHECK: %struct.A = type { float addrspace(4)* }
// %struct.testFieldAlignment = type { i32 addrspace(4)*, [8 x i8], [4 x float], i32, [12 x i8] }
// CHECK: %[[WRAPPER_LAMBDA_PTR:[a-zA-Z0-9_.]+]] = type { float addrspace(1)* }
// CHECK: define {{.*}}spir_kernel void @{{.*}}structs
// CHECK-SAME: %[[GENERATED_B]]* noundef byval(%[[GENERATED_B]]) align 16 %_arg_Obj1
// CHECK-SAME: %[[GENERATED_TESTFIELDALIGNMENT]]* noundef byval(%[[GENERATED_TESTFIELDALIGNMENT]]) align 16 %_arg_Obj2
// CHECK: define {{.*}}spir_kernel void @{{.*}}lambdas{{.*}}(%[[WRAPPER_LAMBDA_PTR]]* noundef byval(%[[WRAPPER_LAMBDA_PTR]]) align 8 %_arg_Lambda)
