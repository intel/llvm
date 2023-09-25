; RUN: opt -fpbuiltin-fn-selection -S < %s | FileCheck %s

; CHECK: call float @sin(float
define void @test_scalar_inexact(float %f1) {
  %t7 = call float @llvm.fpbuiltin.sin.f32(float %f1) #0
  ret void
}
declare float @llvm.fpbuiltin.sin.f32(float)
attributes #0 = { "fpbuiltin-max-error"="0.6" }

