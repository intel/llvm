// RUN: %clang_cc1 -triple spir64-unknown-unknown -x cl -cl-std=CL2.0 -O0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv -spirv-ext=+SPV_INTEL_inline_assembly %t.bc -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o %t.bc
// RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV: {{[0-9]+}} Capability AsmINTEL
// CHECK-SPIRV: {{[0-9]+}} Extension "SPV_INTEL_inline_assembly"
// CHECK-SPIRV: {{[0-9]+}} AsmTargetINTEL
// CHECK-SPIRV: {{[0-9]+}} AsmINTEL

// CHECK-LLVM: call void asm sideeffect

kernel void foo() {
  __asm__ volatile ("");
}
