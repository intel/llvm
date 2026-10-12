; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define <4 x float> @reverse(<4 x float> %a) {
; CHECK-LABEL: @reverse(
; CHECK-NEXT:    [[R:%.*]] = call <4 x float> @llvm.vector.reverse.v4f32(<4 x float> [[A:%.*]])
; CHECK-NEXT:    [[S:%.*]] = call <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float> [[R]]) #[[ATTR:[0-9]+]]
; CHECK-NEXT:    ret <4 x float> [[S]]
  %r = call <4 x float> @llvm.vector.reverse.v4f32(<4 x float> %a)
  %s = call <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float> %r) #0
  ret <4 x float> %s
}

; CHECK: attributes #[[ATTR]] = { "fpbuiltin-max-error"="1.0" }

attributes #0 = { "fpbuiltin-max-error"="1.0" }
