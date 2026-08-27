; RUN: not llvm-spirv %s -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidLlvmModule: Invalid LLVM module: Entry point function must have a name

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @0() {
  ret void
}
