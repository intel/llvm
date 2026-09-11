; RUN: not llvm-spirv %s -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidLlvmModule: Invalid LLVM module: Module-level inline assembly is not supported in SPIR-V

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

module asm "foo"

define spir_kernel void @test() {
  ret void
}
