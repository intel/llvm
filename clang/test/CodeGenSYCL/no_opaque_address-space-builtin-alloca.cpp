// RUN: %clang_cc1 -triple spir64-unknown-linux -fsycl-is-device -disable-llvm-passes -no-opaque-pointers -emit-llvm -x c++ %s -o - | FileCheck %s

// Test to verify that address space cast is generated correctly for __builtin_alloca

__attribute__((sycl_device)) void foo() {
  // CHECK: %TestVar = alloca i32 addrspace(4)*, align 8
  // CHECK: %TestVar.ascast = addrspacecast i32 addrspace(4)** %TestVar to i32 addrspace(4)* addrspace(4)*
  // CHECK: %[[ALLOCA:[0-9]+]] = alloca i8, i64 1, align 8
  // CHECK: %[[ADDRSPCAST:[0-9]+]] = addrspacecast i8* %[[ALLOCA]] to i8 addrspace(4)*
  // CHECK: %[[BITCAST:[0-9]+]] = bitcast i8 addrspace(4)* %[[ADDRSPCAST]] to i32 addrspace(4)*
  // CHECK: store i32 addrspace(4)* %[[BITCAST]], i32 addrspace(4)* addrspace(4)* %TestVar.ascast, align 8
  int *TestVar = (int *)__builtin_alloca(1);
}
