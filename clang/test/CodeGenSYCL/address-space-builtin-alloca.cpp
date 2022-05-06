// RUN: %clang_cc1 -triple spir64-unknown-linux -fsycl-is-device -disable-llvm-passes -opaque-pointers -emit-llvm -x c++ %s -o - | FileCheck %s

// Test to verify that address space cast is generated correctly for __builtin_alloca

__attribute__((sycl_device)) void foo() {
  // CHECK: %TestVar = alloca ptr addrspace(4), align 8
  // CHECK: %TestVar.ascast = addrspacecast ptr %TestVar to ptr addrspace(4)
  // CHECK: %[[ALLOCA:[0-9]+]] = alloca i8, i64 1, align 8
  // CHECK: %[[ADDRSPCAST:[0-9]+]] = addrspacecast ptr %[[ALLOCA]] to ptr addrspace(4)
  // CHECK: store ptr addrspace(4) %[[ADDRSPCAST]], ptr addrspace(4) %TestVar.ascast, align 8
  int *TestVar = (int *)__builtin_alloca(1);
}
