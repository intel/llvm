; Test checks if @llvm.fpbuiltin.fdiv and @llvm.fpbuiltin.sqrt remain if
; other fpbuiltin intrinsic is used in the module.

; RUN: opt -passes=sycl-sqrt-fdiv-max-error-clean-up < %s -S | FileCheck %s

; CHECK: llvm.fpbuiltin.fdiv.f32
; CHECK: llvm.fpbuiltin.sqrt.f32
; CHECK: fpbuiltin-max-error

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define void @test_fp_max_error_decoration(float %f1, float %f2) {
entry:
  %v1 = call float @llvm.fpbuiltin.fdiv.f32(float %f1, float %f2) #0
  %v2 = call float @llvm.fpbuiltin.sqrt.f32(float %v1) #1
  %v3 = call float @llvm.fpbuiltin.exp.f32(float %v2)
  ret void
}

declare float @llvm.fpbuiltin.fdiv.f32(float, float)

declare float @llvm.fpbuiltin.sqrt.f32(float)

declare float @llvm.fpbuiltin.exp.f32(float)

attributes #0 = { "fpbuiltin-max-error"="2.0" }
attributes #1 = { "fpbuiltin-max-error"="3.0" }
