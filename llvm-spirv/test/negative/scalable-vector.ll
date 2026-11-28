; Check that the translator reports an error for a scalable vector type.

; RUN: not llvm-spirv %s -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidLlvmModule: Invalid LLVM module: LLVM module contains scalable vector type, which doesn't have a counterpart in SPIR-V

target triple = "spir64-unknown-unknown"

define spir_kernel void @k() {
  %a = alloca <vscale x 4 x i32>
  ret void
}
