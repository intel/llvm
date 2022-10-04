// RUN: %clang_cc1 -internal-isystem Inputs -fsycl-is-device -std=c++20 -triple spir64-unknown-unknown -disable-llvm-passes -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

#include "Inputs/sycl.hpp"

// This test checks that we correctly capture binding declarations.

void foo() {
  sycl::handler h;
  int a[2] = {1, 2};
  auto [x, y] = a;
  struct S {
    float b[3] = { 0, 3.0f, 4.0 };
  } s;
  auto [f1, f2, f3] = s.b;
  auto Lambda = [=]() { x = 10; f2 = 2.3f; };
  h.single_task(Lambda);
}

// CHECK: %class.anon = type { i32, float }

// Check the sycl kernel arguments - one int and one float parameter
// CHECK: define {{.*}} spir_kernel void @{{.*}}foov{{.*}}(i32 {{.*}} %_arg_, float {{.*}} %_arg_1)
// CHECK: entry:

// Check alloca of the captured types
// CHECK: %_arg_.addr = alloca i32, align 4
// CHECK: %_arg_.addr2 = alloca float, align 4
// CHECK: %__SYCLKernel = alloca %class.anon, align 4

// Copy the parameters into the alloca-ed addresses
// CHECK: store i32 %_arg_, i32 addrspace(4)* %_arg_.addr
// CHECK: store float %_arg_1, float addrspace(4)* %_arg_.addr2

// Store the int and the float into the struct created
// CHECK: %1 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %__SYCLKernel{{.*}}, i32 0, i32 0
// CHECK: %2 = load i32, i32 addrspace(4)* %_arg_.addr
// CHECK: store i32 %2, i32 addrspace(4)* %1
// CHECK: %3 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %__SYCLKernel{{.*}}, i32 0, i32 1
// CHECK: %4 = load float, float addrspace(4)* %_arg_.addr2
// CHECK: store float %4, float addrspace(4)* %3

// Call the lambda
// CHECK: call spir_func void @{{.*}}foo{{.*}}(%class.anon addrspace(4)* {{.*}} %__SYCLKernel{{.*}})
// CHECK:   ret void

// Check the lambda call
// CHECK: define {{.*}} spir_func void @{{.*}}foo{{.*}}(%class.anon addrspace(4)* {{.*}} %this)
// CHECK: entry:
// CHECK:  %this.addr = alloca %class.anon addrspace(4)*
// CHECK:  %this.addr.ascast = addrspacecast %class.anon addrspace(4)** %this.addr to %class.anon addrspace(4)* addrspace(4)*
// CHECK:  store %class.anon addrspace(4)* %this, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast
// CHECK:  %this1 = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast

// Check the store of 10 into the int value
// CHECK:  %0 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this1, i32 0, i32 0
// CHECK:  store i32 10, i32 addrspace(4)* %0

// Check the store of 2.3f into the float value
// CHECK:  %1 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this1, i32 0, i32 1
// CHECK:  store float 0x4002666660000000, float addrspace(4)* %1
// CHECK:  ret void
