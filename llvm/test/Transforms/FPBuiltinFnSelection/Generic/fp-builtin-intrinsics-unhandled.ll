; RUN: not opt -alt-math-library=test -fpbuiltin-fn-selection -S < %s 2>&1 | FileCheck %s

; Basic test for fp-builtin intrinsics error handling when the callsite
; contains an unrecognized fp attribute.

; CHECK: LLVM ERROR: llvm.fpbuiltin.cos.f32 was called with unrecognized floating-point attributes

define void @test_scalar_cr(float %f) {
entry:
  %t1 = call float @llvm.fpbuiltin.cos.f32(float %f) #0
  ret void
}

declare float @llvm.fpbuiltin.cos.f32(float)

attributes #0 = { "fpbuiltin-unknown"="true" }
