// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

kernel void testConvertPtrToU(global int *a, global unsigned long *res) {
  res[0] = (unsigned long)&a[0];
}

// CHECK-SPIRV: 4 ConvertPtrToU

// CHECK-LLVM-LABEL: @testConvertPtrToU
// CHECK-LLVM: %0 = ptrtoint ptr addrspace(1) %a to i32
// CHECK-LLVM: zext i32 %0 to i64

kernel void testConvertUToPtr(unsigned long a) {
  global unsigned int *res = (global unsigned int *)a;
  res[0] = 0;
}

// CHECK-SPIRV: 4 ConvertUToPtr

// CHECK-LLVM-LABEL: @testConvertUToPtr
// CHECK-LLVM: %[[Conv:[a-z]+]] = trunc i64 %a to i32
// CHECK-LLVM: inttoptr i32 %[[Conv]] to ptr addrspace(1)
