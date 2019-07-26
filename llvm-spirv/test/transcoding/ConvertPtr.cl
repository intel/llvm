// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

kernel void test(global int *a, global unsigned long *res) {
  res[0] = (unsigned long)&a[0];
}

// CHECK-SPIRV: ConvertPtrToU

// CHECK-LLVM-LABEL: @test
// CHECK-LLVM: %0 = ptrtoint i32 addrspace(1)* %a to i32
// CHECK-LLVM: zext i32 %0 to i64
