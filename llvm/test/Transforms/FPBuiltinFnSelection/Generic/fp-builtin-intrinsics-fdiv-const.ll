; RUN: opt -fpbuiltin-fn-selection -S < %s | FileCheck %s

; llvm.fpbuiltin.fdiv by a constant divisor with an exact FP reciprocal
; should lower to a multiply by the reciprocal, matching what InstCombine
; would do for a plain fdiv (see CMPLRLLVM-77622).

; CHECK-LABEL: @test_fdiv_exact_reciprocal
; CHECK: %{{.*}} = fmul float %x, 2.500000e-01
; CHECK-NOT: fdiv
define float @test_fdiv_exact_reciprocal(float %x) {
entry:
  %r = call float @llvm.fpbuiltin.fdiv.f32(float %x, float 4.0) #0
  ret float %r
}

; CHECK-LABEL: @test_fdiv_inexact_reciprocal
; CHECK: %{{.*}} = fdiv float %x, 3.000000e+00
define float @test_fdiv_inexact_reciprocal(float %x) {
entry:
  %r = call float @llvm.fpbuiltin.fdiv.f32(float %x, float 3.0) #0
  ret float %r
}

; CHECK-LABEL: @test_fdiv_nonconstant_divisor
; CHECK: %{{.*}} = fdiv float %x, %y
define float @test_fdiv_nonconstant_divisor(float %x, float %y) {
entry:
  %r = call float @llvm.fpbuiltin.fdiv.f32(float %x, float %y) #0
  ret float %r
}

declare float @llvm.fpbuiltin.fdiv.f32(float, float)

attributes #0 = { "fpbuiltin-max-error"="2.5" }
