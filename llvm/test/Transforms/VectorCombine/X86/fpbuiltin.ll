; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -passes=vector-combine -S | FileCheck %s

define <4 x float> @shuffle_of_calls(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: @shuffle_of_calls(
; CHECK-NEXT:    [[X:%.*]] = call <2 x float> @llvm.fpbuiltin.sqrt.v2f32(<2 x float> [[A:%.*]]) #[[ATTR0:[0-9]+]]
; CHECK-NEXT:    [[Y:%.*]] = call <2 x float> @llvm.fpbuiltin.sqrt.v2f32(<2 x float> [[B:%.*]]) #[[ATTR1:[0-9]+]]
; CHECK-NEXT:    [[S:%.*]] = shufflevector <2 x float> [[X]], <2 x float> [[Y]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    ret <4 x float> [[S]]
  %x = call <2 x float> @llvm.fpbuiltin.sqrt.v2f32(<2 x float> %a) #0
  %y = call <2 x float> @llvm.fpbuiltin.sqrt.v2f32(<2 x float> %b) #1
  %s = shufflevector <2 x float> %x, <2 x float> %y, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %s
}

define <4 x float> @permute_of_call(<4 x float> %a) {
; CHECK-LABEL: @permute_of_call(
; CHECK-NEXT:    [[X:%.*]] = call <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float> [[A:%.*]]) #[[ATTR0]]
; CHECK-NEXT:    [[S:%.*]] = shufflevector <4 x float> [[X]], <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT:    ret <4 x float> [[S]]
  %x = call <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float> %a) #0
  %s = shufflevector <4 x float> %x, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %s
}

; CHECK: attributes #[[ATTR0]] = { "fpbuiltin-max-error"="1.0" }
; CHECK: attributes #[[ATTR1]] = { "fpbuiltin-max-error"="4.0" }

attributes #0 = { "fpbuiltin-max-error"="1.0" }
attributes #1 = { "fpbuiltin-max-error"="4.0" }
