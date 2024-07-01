// Ensure __builtin_printf is translated to a SPIRV printf instruction

// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// Change TODO to RUN when spirv-val allows array of 8-bit ints for format
// TODO: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] printf [[#]]
// CHECK-LLVM: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) {{.*}})

kernel void BuiltinPrintf() {
  __builtin_printf("Hello World");
}


