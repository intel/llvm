; RUN: opt -alt-math-library=test -fpbuiltin-fn-selection -S < %s | FileCheck %s

; Basic argument tests for fp-builtin intrinsics.
; Only a few representative functions are tested.

; CHECK-LABEL: @test_scalar_cr
; CHECK: call half @__test_altmath_sqrth_cr
; CHECK: call half @__test_altmath_rsqrth_cr
; CHECK: call float @__test_altmath_sinf_cr
; CHECK: call float @__test_altmath_sqrtf_cr
; CHECK: call float @__test_altmath_rsqrtf_cr
; CHECK: call double @__test_altmath_sin_cr
; CHECK: call double @__test_altmath_sqrt_cr
; CHECK: call double @__test_altmath_rsqrt_cr
define void @test_scalar_cr(half %h, float %f, double %d) {
entry:
  %t1 = call half @llvm.fpbuiltin.sqrt.f16(half %h) #0
  %t2 = call half @llvm.fpbuiltin.rsqrt.f16(half %h) #0
  %t3 = call float @llvm.fpbuiltin.sin.f32(float %f) #0
  %t4 = call float @llvm.fpbuiltin.sqrt.f32(float %f) #0
  %t5 = call float @llvm.fpbuiltin.rsqrt.f32(float %f) #0
  %t6 = call double @llvm.fpbuiltin.sin.f64(double %d) #0
  %t7 = call double @llvm.fpbuiltin.sqrt.f64(double %d) #0
  %t8 = call double @llvm.fpbuiltin.rsqrt.f64(double %d) #0
  ret void
}

; CHECK-LABEL: @test_scalar_1_0
; CHECK: call half @__test_altmath_sinh_high
; CHECK: call half @__test_altmath_cosh_high
; CHECK: call float @__test_altmath_sinf_high
; CHECK: call float @__test_altmath_cosf_high
; CHECK: call float @__test_altmath_tanf_high
; CHECK: call float @__test_altmath_rsqrtf_high
; CHECK: call double @__test_altmath_sin_high
; CHECK: call double @__test_altmath_cos_high
; CHECK: call double @__test_altmath_tan_high
; CHECK: call double @__test_altmath_rsqrt_high
define void @test_scalar_1_0(half %h, float %f, double %d) {
entry:
  %t1 = call half @llvm.fpbuiltin.sin.f16(half %h) #1
  %t2 = call half @llvm.fpbuiltin.cos.f16(half %h) #1
  %t3 = call float @llvm.fpbuiltin.sin.f32(float %f) #1
  %t4 = call float @llvm.fpbuiltin.cos.f32(float %f) #1
  %t5 = call float @llvm.fpbuiltin.tan.f32(float %f) #1
  %t6 = call float @llvm.fpbuiltin.rsqrt.f32(float %f) #1
  %t7 = call double @llvm.fpbuiltin.sin.f64(double %d) #1
  %t8 = call double @llvm.fpbuiltin.cos.f64(double %d) #1
  %t9 = call double @llvm.fpbuiltin.tan.f64(double %d) #1
  %t10 = call double @llvm.fpbuiltin.rsqrt.f64(double %d) #1
  ret void
}

; CHECK-LABEL: @test_scalar_2_5
; CHECK: call half @__test_altmath_fdivh_med
; CHECK: call float @__test_altmath_fdivf_med
; CHECK: call float @__test_altmath_sqrtf_med
; CHECK: call double @__test_altmath_fdiv_med
; CHECK: call double @__test_altmath_sqrt_med
define void @test_scalar_2_5(half %h1, half %h2, float %f1, float %f2,
                            double %d1, double %d2) {
entry:
  %t1 = call half @llvm.fpbuiltin.fdiv.f16(half %h1, half %h2) #2
  %t2 = call float @llvm.fpbuiltin.fdiv.f32(float %f1, float %f2) #2
  %t3 = call float @llvm.fpbuiltin.sqrt.f32(float %f1) #2
  %t4 = call double @llvm.fpbuiltin.fdiv.f64(double %d1, double %d2) #2
  %t5 = call double @llvm.fpbuiltin.sqrt.f64(double %d1) #2
  ret void
}

; CHECK-LABEL: @test_scalar_4_0
; CHECK: call half @__test_altmath_cosh_med
; CHECK: call float @__test_altmath_cosf_med
; CHECK: call double @__test_altmath_cos_med
define void @test_scalar_4_0(half %h, float %f, double %d) {
entry:
  %t1 = call half @llvm.fpbuiltin.cos.f16(half %h) #3
  %t2 = call float @llvm.fpbuiltin.cos.f32(float %f) #3
  %t3 = call double @llvm.fpbuiltin.cos.f64(double %d) #3
  ret void
}

; CHECK-LABEL: @test_scalar_4096
; CHECK: call float @__test_altmath_rsqrtf_low
; CHECK: call double @__test_altmath_rsqrt_low
define void @test_scalar_4096(float %f, double %d) {
entry:
  %t6 = call float @llvm.fpbuiltin.rsqrt.f32(float %f) #4
  %t10 = call double @llvm.fpbuiltin.rsqrt.f64(double %d) #4
  ret void
}

; CHECK-LABEL: @test_vector_1_0
; CHECK: call <4 x float> @__test_altmath_sinf4_high
; CHECK: call <4 x float> @__test_altmath_cosf4_high
; CHECK: call <8 x float> @__test_altmath_sinf8_high
; CHECK: call <8 x float> @__test_altmath_cosf8_high
; CHECK: call <2 x double> @__test_altmath_sin2_high
; CHECK: call <2 x double> @__test_altmath_cos2_high
define void @test_vector_1_0(<4 x float> %v4f, <8 x float> %v8f, <2 x double> %vd) {
entry:
  %t1 = call <4 x float> @llvm.fpbuiltin.sin.v4f32(<4 x float> %v4f) #1
  %t2 = call <4 x float> @llvm.fpbuiltin.cos.v4f32(<4 x float> %v4f) #1
  %t3 = call <8 x float> @llvm.fpbuiltin.sin.v8f32(<8 x float> %v8f) #1
  %t4 = call <8 x float> @llvm.fpbuiltin.cos.v8f32(<8 x float> %v8f) #1
  %t5 = call <2 x double> @llvm.fpbuiltin.sin.v2f64(<2 x double> %vd) #1
  %t6 = call <2 x double> @llvm.fpbuiltin.cos.v2f64(<2 x double> %vd) #1
  ret void
}

; TODO: Add a test with different vector sizes of the same base type


; Test cases where the only available implementations are more accurate than
;   the required accuracy (3.5)
; CHECK-LABEL: @test_scalar_inexact
; CHECK: call half @__test_altmath_fdivh_med
; CHECK: call half @__test_altmath_sinh_high
; CHECK: call half @__test_altmath_cosh_high
; CHECK: call half @__test_altmath_sqrth_cr
; CHECK: call half @__test_altmath_rsqrth_cr
; CHECK: call float @__test_altmath_fdivf_med
; CHECK: call float @__test_altmath_sinf_high
; CHECK: call float @__test_altmath_cosf_high
; CHECK: call float @__test_altmath_tanf_high
; CHECK: call float @__test_altmath_sqrtf_med
; CHECK: call float @__test_altmath_rsqrtf_high
; CHECK: call double @__test_altmath_fdiv_med
; CHECK: call double @__test_altmath_sin_high
; CHECK: call double @__test_altmath_cos_high
; CHECK: call double @__test_altmath_tan_high
; CHECK: call double @__test_altmath_sqrt_med
; CHECK: call double @__test_altmath_rsqrt_high
define void @test_scalar_inexact(half %h1, half %h2, float %f1, float %f2,
                                 double %d1, double %d2) {
entry:
  %t1 = call half @llvm.fpbuiltin.fdiv.f16(half %h1, half %h2) #5
  %t2 = call half @llvm.fpbuiltin.sin.f16(half %h1) #5
  %t3 = call half @llvm.fpbuiltin.cos.f16(half %h1) #5
  %t4 = call half @llvm.fpbuiltin.sqrt.f16(half %h1) #5
  %t5 = call half @llvm.fpbuiltin.rsqrt.f16(half %h1) #5
  %t6 = call float @llvm.fpbuiltin.fdiv.f32(float %f1, float %f2) #5
  %t7 = call float @llvm.fpbuiltin.sin.f32(float %f1) #5
  %t8 = call float @llvm.fpbuiltin.cos.f32(float %f1) #5
  %t9 = call float @llvm.fpbuiltin.tan.f32(float %f1) #5
  %t10 = call float @llvm.fpbuiltin.sqrt.f32(float %f1) #5
  %t11 = call float @llvm.fpbuiltin.rsqrt.f32(float %f1) #5
  %t12 = call double @llvm.fpbuiltin.fdiv.f64(double %d1, double %d2) #5
  %t13 = call double @llvm.fpbuiltin.sin.f64(double %d1) #5
  %t14 = call double @llvm.fpbuiltin.cos.f64(double %d1) #5
  %t15 = call double @llvm.fpbuiltin.tan.f64(double %d1) #5
  %t16 = call double @llvm.fpbuiltin.sqrt.f64(double %d1) #5
  %t17 = call double @llvm.fpbuiltin.rsqrt.f64(double %d1) #5
  ret void
}

declare half @llvm.fpbuiltin.fdiv.f16(half, half)
declare half @llvm.fpbuiltin.sin.f16(half)
declare half @llvm.fpbuiltin.cos.f16(half)
declare half @llvm.fpbuiltin.sqrt.f16(half)
declare half @llvm.fpbuiltin.rsqrt.f16(half)
declare float @llvm.fpbuiltin.fdiv.f32(float, float)
declare float @llvm.fpbuiltin.sin.f32(float)
declare float @llvm.fpbuiltin.cos.f32(float)
declare float @llvm.fpbuiltin.tan.f32(float)
declare float @llvm.fpbuiltin.sqrt.f32(float)
declare float @llvm.fpbuiltin.rsqrt.f32(float)
declare double @llvm.fpbuiltin.fdiv.f64(double, double)
declare double @llvm.fpbuiltin.sin.f64(double)
declare double @llvm.fpbuiltin.cos.f64(double)
declare double @llvm.fpbuiltin.tan.f64(double)
declare double @llvm.fpbuiltin.sqrt.f64(double)
declare double @llvm.fpbuiltin.rsqrt.f64(double)
declare <4 x float> @llvm.fpbuiltin.sin.v4f32(<4 x float>)
declare <4 x float> @llvm.fpbuiltin.cos.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.sin.v8f32(<8 x float>)
declare <8 x float> @llvm.fpbuiltin.cos.v8f32(<8 x float>)
declare <2 x double> @llvm.fpbuiltin.sin.v2f64(<2 x double>)
declare <2 x double> @llvm.fpbuiltin.cos.v2f64(<2 x double>)

attributes #0 = { "fpbuiltin-max-error"="0.5" }
attributes #1 = { "fpbuiltin-max-error"="1.0" }
attributes #2 = { "fpbuiltin-max-error"="2.5" }
attributes #3 = { "fpbuiltin-max-error"="4.0" }
attributes #4 = { "fpbuiltin-max-error"="4096.0" }
attributes #5 = { "fpbuiltin-max-error"="3.0" }
