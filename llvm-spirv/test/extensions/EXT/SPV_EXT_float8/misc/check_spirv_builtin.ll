; This test checks, that function with __builtin_spirv placed in the middle of
; the name is not translated as internal builtin.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: _Z19boo__builtin_spirv_fs
; CHECK-LLVM: _Z19boo__builtin_spirv_fs

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind readnone
define spir_func void @foo() {
entry:
  %0 = call spir_func half @_Z19boo__builtin_spirv_fs(float 1.0, i16 4)
  ret void
}

declare dso_local spir_func half @_Z19boo__builtin_spirv_fs(float, i16)
