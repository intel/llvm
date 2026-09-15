; RUN: llvm-spirv %s -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; A numeric-named function has no meaningful linkage name, so no
; LinkageAttributes decoration should be emitted for it.
; CHECK-NOT: LinkageAttributes

define spir_func void @0() {
  ret void
}

define spir_kernel void @kernel() {
  call spir_func void @0()
  ret void
}
