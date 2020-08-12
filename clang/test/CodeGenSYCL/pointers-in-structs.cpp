// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that compiler generates correct address spaces for pointer
// kernel arguments that are wrapped by struct.

#include <sycl.hpp>

struct A {
  float *F;
};

struct B {
  int *F1;
  float *F2;
  A F3;
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

// CHECK: define spir_kernel void @{{.*}}structs{{.*}}(i32 addrspace(1)* %_arg_F1, float addrspace(1)* %_arg_F2, float addrspace(1)* %_arg_F)
// CHECK: define spir_kernel void @{{.*}}lambdas{{.*}}(float addrspace(1)* %_arg_)
