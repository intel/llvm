; RUN: opt -LowerESIMD -S < %s | FileCheck %s

; This test checks that LowerESIMD pass correctly lowers some llvm intrinsics
; which can't be handled by the VC BE.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

declare float @llvm.fmuladd.f32(float %x, float %y, float %z)
declare double @llvm.fmuladd.f64(double %x, double %y, double %z)

define spir_func float @test_fmuladd_f32(float %x, float %y, float %z) {
  %1 = call float @llvm.fmuladd.f32(float %x, float %y, float %z)
; CHECK: %[[A:[0-9a-zA-Z\._]+]] = fmul float %x, %y
; CHECK: %[[B:[0-9a-zA-Z\._]+]] = fadd float %[[A]], %z
  ret float %1
; CHECK: ret float %[[B]]
}

define spir_func double @test_fmuladd_f64(double %x, double %y, double %z) {
  %1 = call double @llvm.fmuladd.f64(double %x, double %y, double %z)
; CHECK: %[[A:[0-9a-zA-Z\._]+]] = fmul double %x, %y
; CHECK: %[[B:[0-9a-zA-Z\._]+]] = fadd double %[[A]], %z
  ret double %1
; CHECK: ret double %[[B]]
}

