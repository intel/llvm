; RUN: not llvm-spirv %s -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidFunctionCall
; CHECK: llvm.lround.i1.f32

target triple = "spir64-unknown-unknown"

define spir_func i1 @f(float %x) {
  %r = call i1 @llvm.lround.i1.f32(float %x)
  ret i1 %r
}
