// RUN: %clang_cc1 -O0 -S -triple spir-unknown-unknown -cl-std=CL2.0 -x cl -disable-O0-optnone %s -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv -s %t.bc
// RUN: llvm-dis < %t.bc | FileCheck %s --check-prefixes=CHECK-WO
// RUN: llvm-spirv -s -spirv-mem2reg %t.bc -o %t.opt.bc
// RUN: llvm-dis < %t.opt.bc | FileCheck %s --check-prefixes=CHECK-W
// CHECK-W-LABEL: spir_func void @foo
// CHECK-W-NOT: alloca i32
// CHECK-WO-LABEL: spir_kernel void @foo
// CHECK-WO: alloca i32
__kernel void foo(__global int *a) {
    *a = *a + 1;
}
