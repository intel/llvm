// RUN: %clang_cc1 -O0 -S -triple spir-unknown-unknown -cl-std=CL2.0 -x cl -disable-O0-optnone %s -emit-llvm-bc -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers -s %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-dis -opaque-pointers < %t.bc | FileCheck %s --check-prefixes=CHECK-WO
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers -s -spirv-mem2reg %t.bc -o %t.opt.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-dis -opaque-pointers < %t.opt.bc | FileCheck %s --check-prefixes=CHECK-W
// CHECK-W-LABEL: spir_kernel void @foo
// CHECK-W-NOT: alloca
// CHECK-WO-LABEL: spir_kernel void @foo
// CHECK-WO: alloca
__kernel void foo(__global int *a) {
    *a = *a + 1;
}
