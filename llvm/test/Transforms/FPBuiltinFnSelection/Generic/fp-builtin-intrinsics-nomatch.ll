; RUN: not opt -alt-math-library=test -fpbuiltin-fn-selection -S < %s 2>&1 | FileCheck %s

; Basic test for fp-builtin intrinsics error handling when no suitable
; implementation is available.

; CHECK: LLVM ERROR: llvm.fpbuiltin.cos.f32 was called with required accuracy = 0.50 but no suitable implementation was found.

define void @test_scalar_cr(float %f) {
entry:
  %t1 = call float @llvm.fpbuiltin.cos.f32(float %f) #0
  ret void
}

declare float @llvm.fpbuiltin.cos.f32(float)

attributes #0 = { "fpbuiltin-max-error"="0.5" }
