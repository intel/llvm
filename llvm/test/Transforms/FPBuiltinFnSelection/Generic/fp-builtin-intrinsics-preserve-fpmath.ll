; RUN: opt  -fpbuiltin-fn-selection -S < %s 2>&1 | FileCheck %s

; Check that "fpbuiltin-max-error" attribute value is copied to fpmath metadata

; CHECK: %[[#]] = call float @llvm.sqrt.f32(float %{{.*}}), !fpmath ![[#MD:]]
; CHECK: ![[#MD]] = !{float 5.000000e-01}

define void @test_scalar_cr(float %f) {
entry:
  %t1 = call float @llvm.fpbuiltin.sqrt.f32(float %f) #0
  ret void
}

declare float @llvm.fpbuiltin.sqrt.f32(float)

attributes #0 = { "fpbuiltin-max-error"="0.5" }
