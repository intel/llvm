; RUN: opt -alt-math-library=svml -fpbuiltin-fn-selection -S < %s | FileCheck %s

; This test is to verify if all llvm.fpbuiltin.* intrinsic calls can be
; selected to corresponding SVML library calls under different values of
; fpbuiltin-max-error in attribute.

; CHECK-LABEL: @svml_sin
; CHECK: call float @__svml_sinf1_ha(
; CHECK: call float @__svml_sinf1(
; CHECK: call float @__svml_sinf1_ep(
; CHECK: call <4 x float> @__svml_sinf4_ha(
; CHECK: call <4 x float> @__svml_sinf4(
; CHECK: call <4 x float> @__svml_sinf4_ep(
; CHECK: call <8 x float> @__svml_sinf8_ha(
; CHECK: call <8 x float> @__svml_sinf8(
; CHECK: call <8 x float> @__svml_sinf8_ep(
; CHECK: call <16 x float> @__svml_sinf16_ha(
; CHECK: call <16 x float> @__svml_sinf16(
; CHECK: call <16 x float> @__svml_sinf16_ep(
; CHECK: call double @__svml_sin1_ha(
; CHECK: call double @__svml_sin1(
; CHECK: call double @__svml_sin1_ep(
; CHECK: call <2 x double> @__svml_sin2_ha(
; CHECK: call <2 x double> @__svml_sin2(
; CHECK: call <2 x double> @__svml_sin2_ep(
; CHECK: call <4 x double> @__svml_sin4_ha(
; CHECK: call <4 x double> @__svml_sin4(
; CHECK: call <4 x double> @__svml_sin4_ep(
; CHECK: call <8 x double> @__svml_sin8_ha(
; CHECK: call <8 x double> @__svml_sin8(
; CHECK: call <8 x double> @__svml_sin8_ep(
; CHECK: call half @__svml_sins1_ha(
; CHECK: call half @__svml_sins1(
; CHECK: call half @__svml_sins1_ep(
; CHECK: call <4 x half> @__svml_sins4_ha(
; CHECK: call <4 x half> @__svml_sins4(
; CHECK: call <4 x half> @__svml_sins4_ep(
; CHECK: call <8 x half> @__svml_sins8_ha(
; CHECK: call <8 x half> @__svml_sins8(
; CHECK: call <8 x half> @__svml_sins8_ep(
; CHECK: call <16 x half> @__svml_sins16_ha(
; CHECK: call <16 x half> @__svml_sins16(
; CHECK: call <16 x half> @__svml_sins16_ep(
; CHECK: call <32 x half> @__svml_sins32_ha(
; CHECK: call <32 x half> @__svml_sins32(
; CHECK: call <32 x half> @__svml_sins32_ep(
define void @svml_sin(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                      double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                      half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.sin.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.sin.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.sin.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.sin.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.sin.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.sin.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.sin.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.sin.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.sin.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.sin.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.sin.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.sin.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.sin.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.sin.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.sin.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.sin.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.sin.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.sin.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.sin.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.sin.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.sin.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.sin.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.sin.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.sin.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.sin.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.sin.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.sin.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.sin.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.sin.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.sin.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.sin.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.sin.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.sin.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.sin.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.sin.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.sin.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.sin.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.sin.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.sin.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.sin.f32(float)
declare <4 x float> @llvm.fpbuiltin.sin.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.sin.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.sin.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.sin.f64(double)
declare <2 x double> @llvm.fpbuiltin.sin.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.sin.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.sin.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.sin.f16(half)
declare <4 x half> @llvm.fpbuiltin.sin.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.sin.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.sin.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.sin.v32f16(<32 x half>)

; CHECK-LABEL: @svml_sinh
; CHECK: call float @__svml_sinhf1_ha(
; CHECK: call float @__svml_sinhf1(
; CHECK: call float @__svml_sinhf1_ep(
; CHECK: call <4 x float> @__svml_sinhf4_ha(
; CHECK: call <4 x float> @__svml_sinhf4(
; CHECK: call <4 x float> @__svml_sinhf4_ep(
; CHECK: call <8 x float> @__svml_sinhf8_ha(
; CHECK: call <8 x float> @__svml_sinhf8(
; CHECK: call <8 x float> @__svml_sinhf8_ep(
; CHECK: call <16 x float> @__svml_sinhf16_ha(
; CHECK: call <16 x float> @__svml_sinhf16(
; CHECK: call <16 x float> @__svml_sinhf16_ep(
; CHECK: call double @__svml_sinh1_ha(
; CHECK: call double @__svml_sinh1(
; CHECK: call double @__svml_sinh1_ep(
; CHECK: call <2 x double> @__svml_sinh2_ha(
; CHECK: call <2 x double> @__svml_sinh2(
; CHECK: call <2 x double> @__svml_sinh2_ep(
; CHECK: call <4 x double> @__svml_sinh4_ha(
; CHECK: call <4 x double> @__svml_sinh4(
; CHECK: call <4 x double> @__svml_sinh4_ep(
; CHECK: call <8 x double> @__svml_sinh8_ha(
; CHECK: call <8 x double> @__svml_sinh8(
; CHECK: call <8 x double> @__svml_sinh8_ep(
; CHECK: call half @__svml_sinhs1_ha(
; CHECK: call half @__svml_sinhs1(
; CHECK: call half @__svml_sinhs1_ep(
; CHECK: call <4 x half> @__svml_sinhs4_ha(
; CHECK: call <4 x half> @__svml_sinhs4(
; CHECK: call <4 x half> @__svml_sinhs4_ep(
; CHECK: call <8 x half> @__svml_sinhs8_ha(
; CHECK: call <8 x half> @__svml_sinhs8(
; CHECK: call <8 x half> @__svml_sinhs8_ep(
; CHECK: call <16 x half> @__svml_sinhs16_ha(
; CHECK: call <16 x half> @__svml_sinhs16(
; CHECK: call <16 x half> @__svml_sinhs16_ep(
; CHECK: call <32 x half> @__svml_sinhs32_ha(
; CHECK: call <32 x half> @__svml_sinhs32(
; CHECK: call <32 x half> @__svml_sinhs32_ep(
define void @svml_sinh(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.sinh.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.sinh.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.sinh.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.sinh.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.sinh.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.sinh.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.sinh.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.sinh.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.sinh.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.sinh.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.sinh.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.sinh.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.sinh.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.sinh.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.sinh.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.sinh.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.sinh.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.sinh.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.sinh.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.sinh.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.sinh.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.sinh.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.sinh.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.sinh.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.sinh.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.sinh.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.sinh.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.sinh.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.sinh.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.sinh.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.sinh.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.sinh.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.sinh.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.sinh.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.sinh.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.sinh.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.sinh.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.sinh.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.sinh.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.sinh.f32(float)
declare <4 x float> @llvm.fpbuiltin.sinh.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.sinh.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.sinh.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.sinh.f64(double)
declare <2 x double> @llvm.fpbuiltin.sinh.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.sinh.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.sinh.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.sinh.f16(half)
declare <4 x half> @llvm.fpbuiltin.sinh.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.sinh.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.sinh.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.sinh.v32f16(<32 x half>)

; CHECK-LABEL: @svml_cos
; CHECK: call float @__svml_cosf1_ha(
; CHECK: call float @__svml_cosf1(
; CHECK: call float @__svml_cosf1_ep(
; CHECK: call <4 x float> @__svml_cosf4_ha(
; CHECK: call <4 x float> @__svml_cosf4(
; CHECK: call <4 x float> @__svml_cosf4_ep(
; CHECK: call <8 x float> @__svml_cosf8_ha(
; CHECK: call <8 x float> @__svml_cosf8(
; CHECK: call <8 x float> @__svml_cosf8_ep(
; CHECK: call <16 x float> @__svml_cosf16_ha(
; CHECK: call <16 x float> @__svml_cosf16(
; CHECK: call <16 x float> @__svml_cosf16_ep(
; CHECK: call double @__svml_cos1_ha(
; CHECK: call double @__svml_cos1(
; CHECK: call double @__svml_cos1_ep(
; CHECK: call <2 x double> @__svml_cos2_ha(
; CHECK: call <2 x double> @__svml_cos2(
; CHECK: call <2 x double> @__svml_cos2_ep(
; CHECK: call <4 x double> @__svml_cos4_ha(
; CHECK: call <4 x double> @__svml_cos4(
; CHECK: call <4 x double> @__svml_cos4_ep(
; CHECK: call <8 x double> @__svml_cos8_ha(
; CHECK: call <8 x double> @__svml_cos8(
; CHECK: call <8 x double> @__svml_cos8_ep(
; CHECK: call half @__svml_coss1_ha(
; CHECK: call half @__svml_coss1(
; CHECK: call half @__svml_coss1_ep(
; CHECK: call <4 x half> @__svml_coss4_ha(
; CHECK: call <4 x half> @__svml_coss4(
; CHECK: call <4 x half> @__svml_coss4_ep(
; CHECK: call <8 x half> @__svml_coss8_ha(
; CHECK: call <8 x half> @__svml_coss8(
; CHECK: call <8 x half> @__svml_coss8_ep(
; CHECK: call <16 x half> @__svml_coss16_ha(
; CHECK: call <16 x half> @__svml_coss16(
; CHECK: call <16 x half> @__svml_coss16_ep(
; CHECK: call <32 x half> @__svml_coss32_ha(
; CHECK: call <32 x half> @__svml_coss32(
; CHECK: call <32 x half> @__svml_coss32_ep(
define void @svml_cos(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                      double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                      half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.cos.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.cos.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.cos.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.cos.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.cos.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.cos.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.cos.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.cos.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.cos.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.cos.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.cos.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.cos.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.cos.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.cos.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.cos.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.cos.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.cos.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.cos.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.cos.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.cos.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.cos.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.cos.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.cos.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.cos.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.cos.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.cos.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.cos.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.cos.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.cos.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.cos.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.cos.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.cos.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.cos.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.cos.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.cos.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.cos.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.cos.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.cos.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.cos.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.cos.f32(float)
declare <4 x float> @llvm.fpbuiltin.cos.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.cos.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.cos.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.cos.f64(double)
declare <2 x double> @llvm.fpbuiltin.cos.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.cos.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.cos.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.cos.f16(half)
declare <4 x half> @llvm.fpbuiltin.cos.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.cos.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.cos.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.cos.v32f16(<32 x half>)

; CHECK-LABEL: @svml_cosh
; CHECK: call float @__svml_coshf1_ha(
; CHECK: call float @__svml_coshf1(
; CHECK: call float @__svml_coshf1_ep(
; CHECK: call <4 x float> @__svml_coshf4_ha(
; CHECK: call <4 x float> @__svml_coshf4(
; CHECK: call <4 x float> @__svml_coshf4_ep(
; CHECK: call <8 x float> @__svml_coshf8_ha(
; CHECK: call <8 x float> @__svml_coshf8(
; CHECK: call <8 x float> @__svml_coshf8_ep(
; CHECK: call <16 x float> @__svml_coshf16_ha(
; CHECK: call <16 x float> @__svml_coshf16(
; CHECK: call <16 x float> @__svml_coshf16_ep(
; CHECK: call double @__svml_cosh1_ha(
; CHECK: call double @__svml_cosh1(
; CHECK: call double @__svml_cosh1_ep(
; CHECK: call <2 x double> @__svml_cosh2_ha(
; CHECK: call <2 x double> @__svml_cosh2(
; CHECK: call <2 x double> @__svml_cosh2_ep(
; CHECK: call <4 x double> @__svml_cosh4_ha(
; CHECK: call <4 x double> @__svml_cosh4(
; CHECK: call <4 x double> @__svml_cosh4_ep(
; CHECK: call <8 x double> @__svml_cosh8_ha(
; CHECK: call <8 x double> @__svml_cosh8(
; CHECK: call <8 x double> @__svml_cosh8_ep(
; CHECK: call half @__svml_coshs1_ha(
; CHECK: call half @__svml_coshs1(
; CHECK: call half @__svml_coshs1_ep(
; CHECK: call <4 x half> @__svml_coshs4_ha(
; CHECK: call <4 x half> @__svml_coshs4(
; CHECK: call <4 x half> @__svml_coshs4_ep(
; CHECK: call <8 x half> @__svml_coshs8_ha(
; CHECK: call <8 x half> @__svml_coshs8(
; CHECK: call <8 x half> @__svml_coshs8_ep(
; CHECK: call <16 x half> @__svml_coshs16_ha(
; CHECK: call <16 x half> @__svml_coshs16(
; CHECK: call <16 x half> @__svml_coshs16_ep(
; CHECK: call <32 x half> @__svml_coshs32_ha(
; CHECK: call <32 x half> @__svml_coshs32(
; CHECK: call <32 x half> @__svml_coshs32_ep(
define void @svml_cosh(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.cosh.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.cosh.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.cosh.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.cosh.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.cosh.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.cosh.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.cosh.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.cosh.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.cosh.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.cosh.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.cosh.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.cosh.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.cosh.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.cosh.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.cosh.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.cosh.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.cosh.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.cosh.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.cosh.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.cosh.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.cosh.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.cosh.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.cosh.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.cosh.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.cosh.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.cosh.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.cosh.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.cosh.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.cosh.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.cosh.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.cosh.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.cosh.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.cosh.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.cosh.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.cosh.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.cosh.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.cosh.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.cosh.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.cosh.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.cosh.f32(float)
declare <4 x float> @llvm.fpbuiltin.cosh.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.cosh.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.cosh.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.cosh.f64(double)
declare <2 x double> @llvm.fpbuiltin.cosh.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.cosh.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.cosh.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.cosh.f16(half)
declare <4 x half> @llvm.fpbuiltin.cosh.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.cosh.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.cosh.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.cosh.v32f16(<32 x half>)

; CHECK-LABEL: @svml_tan
; CHECK: call float @__svml_tanf1_ha(
; CHECK: call float @__svml_tanf1(
; CHECK: call float @__svml_tanf1_ep(
; CHECK: call <4 x float> @__svml_tanf4_ha(
; CHECK: call <4 x float> @__svml_tanf4(
; CHECK: call <4 x float> @__svml_tanf4_ep(
; CHECK: call <8 x float> @__svml_tanf8_ha(
; CHECK: call <8 x float> @__svml_tanf8(
; CHECK: call <8 x float> @__svml_tanf8_ep(
; CHECK: call <16 x float> @__svml_tanf16_ha(
; CHECK: call <16 x float> @__svml_tanf16(
; CHECK: call <16 x float> @__svml_tanf16_ep(
; CHECK: call double @__svml_tan1_ha(
; CHECK: call double @__svml_tan1(
; CHECK: call double @__svml_tan1_ep(
; CHECK: call <2 x double> @__svml_tan2_ha(
; CHECK: call <2 x double> @__svml_tan2(
; CHECK: call <2 x double> @__svml_tan2_ep(
; CHECK: call <4 x double> @__svml_tan4_ha(
; CHECK: call <4 x double> @__svml_tan4(
; CHECK: call <4 x double> @__svml_tan4_ep(
; CHECK: call <8 x double> @__svml_tan8_ha(
; CHECK: call <8 x double> @__svml_tan8(
; CHECK: call <8 x double> @__svml_tan8_ep(
; CHECK: call half @__svml_tans1_ha(
; CHECK: call half @__svml_tans1(
; CHECK: call half @__svml_tans1_ep(
; CHECK: call <4 x half> @__svml_tans4_ha(
; CHECK: call <4 x half> @__svml_tans4(
; CHECK: call <4 x half> @__svml_tans4_ep(
; CHECK: call <8 x half> @__svml_tans8_ha(
; CHECK: call <8 x half> @__svml_tans8(
; CHECK: call <8 x half> @__svml_tans8_ep(
; CHECK: call <16 x half> @__svml_tans16_ha(
; CHECK: call <16 x half> @__svml_tans16(
; CHECK: call <16 x half> @__svml_tans16_ep(
; CHECK: call <32 x half> @__svml_tans32_ha(
; CHECK: call <32 x half> @__svml_tans32(
; CHECK: call <32 x half> @__svml_tans32_ep(
define void @svml_tan(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                      double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                      half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.tan.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.tan.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.tan.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.tan.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.tan.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.tan.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.tan.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.tan.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.tan.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.tan.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.tan.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.tan.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.tan.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.tan.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.tan.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.tan.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.tan.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.tan.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.tan.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.tan.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.tan.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.tan.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.tan.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.tan.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.tan.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.tan.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.tan.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.tan.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.tan.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.tan.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.tan.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.tan.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.tan.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.tan.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.tan.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.tan.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.tan.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.tan.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.tan.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.tan.f32(float)
declare <4 x float> @llvm.fpbuiltin.tan.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.tan.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.tan.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.tan.f64(double)
declare <2 x double> @llvm.fpbuiltin.tan.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.tan.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.tan.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.tan.f16(half)
declare <4 x half> @llvm.fpbuiltin.tan.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.tan.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.tan.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.tan.v32f16(<32 x half>)

; CHECK-LABEL: @svml_tanh
; CHECK: call float @__svml_tanhf1_ha(
; CHECK: call float @__svml_tanhf1(
; CHECK: call float @__svml_tanhf1_ep(
; CHECK: call <4 x float> @__svml_tanhf4_ha(
; CHECK: call <4 x float> @__svml_tanhf4(
; CHECK: call <4 x float> @__svml_tanhf4_ep(
; CHECK: call <8 x float> @__svml_tanhf8_ha(
; CHECK: call <8 x float> @__svml_tanhf8(
; CHECK: call <8 x float> @__svml_tanhf8_ep(
; CHECK: call <16 x float> @__svml_tanhf16_ha(
; CHECK: call <16 x float> @__svml_tanhf16(
; CHECK: call <16 x float> @__svml_tanhf16_ep(
; CHECK: call double @__svml_tanh1_ha(
; CHECK: call double @__svml_tanh1(
; CHECK: call double @__svml_tanh1_ep(
; CHECK: call <2 x double> @__svml_tanh2_ha(
; CHECK: call <2 x double> @__svml_tanh2(
; CHECK: call <2 x double> @__svml_tanh2_ep(
; CHECK: call <4 x double> @__svml_tanh4_ha(
; CHECK: call <4 x double> @__svml_tanh4(
; CHECK: call <4 x double> @__svml_tanh4_ep(
; CHECK: call <8 x double> @__svml_tanh8_ha(
; CHECK: call <8 x double> @__svml_tanh8(
; CHECK: call <8 x double> @__svml_tanh8_ep(
; CHECK: call half @__svml_tanhs1_ha(
; CHECK: call half @__svml_tanhs1(
; CHECK: call half @__svml_tanhs1_ep(
; CHECK: call <4 x half> @__svml_tanhs4_ha(
; CHECK: call <4 x half> @__svml_tanhs4(
; CHECK: call <4 x half> @__svml_tanhs4_ep(
; CHECK: call <8 x half> @__svml_tanhs8_ha(
; CHECK: call <8 x half> @__svml_tanhs8(
; CHECK: call <8 x half> @__svml_tanhs8_ep(
; CHECK: call <16 x half> @__svml_tanhs16_ha(
; CHECK: call <16 x half> @__svml_tanhs16(
; CHECK: call <16 x half> @__svml_tanhs16_ep(
; CHECK: call <32 x half> @__svml_tanhs32_ha(
; CHECK: call <32 x half> @__svml_tanhs32(
; CHECK: call <32 x half> @__svml_tanhs32_ep(
define void @svml_tanh(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.tanh.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.tanh.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.tanh.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.tanh.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.tanh.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.tanh.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.tanh.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.tanh.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.tanh.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.tanh.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.tanh.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.tanh.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.tanh.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.tanh.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.tanh.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.tanh.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.tanh.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.tanh.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.tanh.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.tanh.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.tanh.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.tanh.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.tanh.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.tanh.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.tanh.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.tanh.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.tanh.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.tanh.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.tanh.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.tanh.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.tanh.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.tanh.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.tanh.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.tanh.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.tanh.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.tanh.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.tanh.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.tanh.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.tanh.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.tanh.f32(float)
declare <4 x float> @llvm.fpbuiltin.tanh.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.tanh.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.tanh.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.tanh.f64(double)
declare <2 x double> @llvm.fpbuiltin.tanh.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.tanh.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.tanh.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.tanh.f16(half)
declare <4 x half> @llvm.fpbuiltin.tanh.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.tanh.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.tanh.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.tanh.v32f16(<32 x half>)

; CHECK-LABEL: @svml_acos
; CHECK: call float @__svml_acosf1_ha(
; CHECK: call float @__svml_acosf1(
; CHECK: call float @__svml_acosf1_ep(
; CHECK: call <4 x float> @__svml_acosf4_ha(
; CHECK: call <4 x float> @__svml_acosf4(
; CHECK: call <4 x float> @__svml_acosf4_ep(
; CHECK: call <8 x float> @__svml_acosf8_ha(
; CHECK: call <8 x float> @__svml_acosf8(
; CHECK: call <8 x float> @__svml_acosf8_ep(
; CHECK: call <16 x float> @__svml_acosf16_ha(
; CHECK: call <16 x float> @__svml_acosf16(
; CHECK: call <16 x float> @__svml_acosf16_ep(
; CHECK: call double @__svml_acos1_ha(
; CHECK: call double @__svml_acos1(
; CHECK: call double @__svml_acos1_ep(
; CHECK: call <2 x double> @__svml_acos2_ha(
; CHECK: call <2 x double> @__svml_acos2(
; CHECK: call <2 x double> @__svml_acos2_ep(
; CHECK: call <4 x double> @__svml_acos4_ha(
; CHECK: call <4 x double> @__svml_acos4(
; CHECK: call <4 x double> @__svml_acos4_ep(
; CHECK: call <8 x double> @__svml_acos8_ha(
; CHECK: call <8 x double> @__svml_acos8(
; CHECK: call <8 x double> @__svml_acos8_ep(
; CHECK: call half @__svml_acoss1_ha(
; CHECK: call half @__svml_acoss1(
; CHECK: call half @__svml_acoss1_ep(
; CHECK: call <4 x half> @__svml_acoss4_ha(
; CHECK: call <4 x half> @__svml_acoss4(
; CHECK: call <4 x half> @__svml_acoss4_ep(
; CHECK: call <8 x half> @__svml_acoss8_ha(
; CHECK: call <8 x half> @__svml_acoss8(
; CHECK: call <8 x half> @__svml_acoss8_ep(
; CHECK: call <16 x half> @__svml_acoss16_ha(
; CHECK: call <16 x half> @__svml_acoss16(
; CHECK: call <16 x half> @__svml_acoss16_ep(
; CHECK: call <32 x half> @__svml_acoss32_ha(
; CHECK: call <32 x half> @__svml_acoss32(
; CHECK: call <32 x half> @__svml_acoss32_ep(
define void @svml_acos(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.acos.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.acos.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.acos.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.acos.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.acos.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.acos.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.acos.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.acos.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.acos.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.acos.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.acos.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.acos.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.acos.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.acos.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.acos.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.acos.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.acos.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.acos.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.acos.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.acos.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.acos.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.acos.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.acos.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.acos.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.acos.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.acos.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.acos.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.acos.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.acos.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.acos.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.acos.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.acos.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.acos.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.acos.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.acos.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.acos.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.acos.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.acos.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.acos.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.acos.f32(float)
declare <4 x float> @llvm.fpbuiltin.acos.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.acos.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.acos.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.acos.f64(double)
declare <2 x double> @llvm.fpbuiltin.acos.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.acos.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.acos.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.acos.f16(half)
declare <4 x half> @llvm.fpbuiltin.acos.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.acos.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.acos.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.acos.v32f16(<32 x half>)

; CHECK-LABEL: @svml_acosh
; CHECK: call float @__svml_acoshf1_ha(
; CHECK: call float @__svml_acoshf1(
; CHECK: call float @__svml_acoshf1_ep(
; CHECK: call <4 x float> @__svml_acoshf4_ha(
; CHECK: call <4 x float> @__svml_acoshf4(
; CHECK: call <4 x float> @__svml_acoshf4_ep(
; CHECK: call <8 x float> @__svml_acoshf8_ha(
; CHECK: call <8 x float> @__svml_acoshf8(
; CHECK: call <8 x float> @__svml_acoshf8_ep(
; CHECK: call <16 x float> @__svml_acoshf16_ha(
; CHECK: call <16 x float> @__svml_acoshf16(
; CHECK: call <16 x float> @__svml_acoshf16_ep(
; CHECK: call double @__svml_acosh1_ha(
; CHECK: call double @__svml_acosh1(
; CHECK: call double @__svml_acosh1_ep(
; CHECK: call <2 x double> @__svml_acosh2_ha(
; CHECK: call <2 x double> @__svml_acosh2(
; CHECK: call <2 x double> @__svml_acosh2_ep(
; CHECK: call <4 x double> @__svml_acosh4_ha(
; CHECK: call <4 x double> @__svml_acosh4(
; CHECK: call <4 x double> @__svml_acosh4_ep(
; CHECK: call <8 x double> @__svml_acosh8_ha(
; CHECK: call <8 x double> @__svml_acosh8(
; CHECK: call <8 x double> @__svml_acosh8_ep(
; CHECK: call half @__svml_acoshs1_ha(
; CHECK: call half @__svml_acoshs1(
; CHECK: call half @__svml_acoshs1_ep(
; CHECK: call <4 x half> @__svml_acoshs4_ha(
; CHECK: call <4 x half> @__svml_acoshs4(
; CHECK: call <4 x half> @__svml_acoshs4_ep(
; CHECK: call <8 x half> @__svml_acoshs8_ha(
; CHECK: call <8 x half> @__svml_acoshs8(
; CHECK: call <8 x half> @__svml_acoshs8_ep(
; CHECK: call <16 x half> @__svml_acoshs16_ha(
; CHECK: call <16 x half> @__svml_acoshs16(
; CHECK: call <16 x half> @__svml_acoshs16_ep(
; CHECK: call <32 x half> @__svml_acoshs32_ha(
; CHECK: call <32 x half> @__svml_acoshs32(
; CHECK: call <32 x half> @__svml_acoshs32_ep(
define void @svml_acosh(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                        double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                        half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.acosh.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.acosh.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.acosh.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.acosh.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.acosh.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.acosh.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.acosh.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.acosh.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.acosh.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.acosh.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.acosh.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.acosh.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.acosh.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.acosh.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.acosh.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.acosh.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.acosh.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.acosh.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.acosh.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.acosh.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.acosh.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.acosh.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.acosh.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.acosh.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.acosh.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.acosh.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.acosh.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.acosh.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.acosh.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.acosh.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.acosh.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.acosh.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.acosh.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.acosh.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.acosh.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.acosh.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.acosh.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.acosh.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.acosh.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.acosh.f32(float)
declare <4 x float> @llvm.fpbuiltin.acosh.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.acosh.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.acosh.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.acosh.f64(double)
declare <2 x double> @llvm.fpbuiltin.acosh.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.acosh.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.acosh.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.acosh.f16(half)
declare <4 x half> @llvm.fpbuiltin.acosh.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.acosh.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.acosh.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.acosh.v32f16(<32 x half>)

; CHECK-LABEL: @svml_asin
; CHECK: call float @__svml_asinf1_ha(
; CHECK: call float @__svml_asinf1(
; CHECK: call float @__svml_asinf1_ep(
; CHECK: call <4 x float> @__svml_asinf4_ha(
; CHECK: call <4 x float> @__svml_asinf4(
; CHECK: call <4 x float> @__svml_asinf4_ep(
; CHECK: call <8 x float> @__svml_asinf8_ha(
; CHECK: call <8 x float> @__svml_asinf8(
; CHECK: call <8 x float> @__svml_asinf8_ep(
; CHECK: call <16 x float> @__svml_asinf16_ha(
; CHECK: call <16 x float> @__svml_asinf16(
; CHECK: call <16 x float> @__svml_asinf16_ep(
; CHECK: call double @__svml_asin1_ha(
; CHECK: call double @__svml_asin1(
; CHECK: call double @__svml_asin1_ep(
; CHECK: call <2 x double> @__svml_asin2_ha(
; CHECK: call <2 x double> @__svml_asin2(
; CHECK: call <2 x double> @__svml_asin2_ep(
; CHECK: call <4 x double> @__svml_asin4_ha(
; CHECK: call <4 x double> @__svml_asin4(
; CHECK: call <4 x double> @__svml_asin4_ep(
; CHECK: call <8 x double> @__svml_asin8_ha(
; CHECK: call <8 x double> @__svml_asin8(
; CHECK: call <8 x double> @__svml_asin8_ep(
; CHECK: call half @__svml_asins1_ha(
; CHECK: call half @__svml_asins1(
; CHECK: call half @__svml_asins1_ep(
; CHECK: call <4 x half> @__svml_asins4_ha(
; CHECK: call <4 x half> @__svml_asins4(
; CHECK: call <4 x half> @__svml_asins4_ep(
; CHECK: call <8 x half> @__svml_asins8_ha(
; CHECK: call <8 x half> @__svml_asins8(
; CHECK: call <8 x half> @__svml_asins8_ep(
; CHECK: call <16 x half> @__svml_asins16_ha(
; CHECK: call <16 x half> @__svml_asins16(
; CHECK: call <16 x half> @__svml_asins16_ep(
; CHECK: call <32 x half> @__svml_asins32_ha(
; CHECK: call <32 x half> @__svml_asins32(
; CHECK: call <32 x half> @__svml_asins32_ep(
define void @svml_asin(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.asin.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.asin.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.asin.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.asin.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.asin.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.asin.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.asin.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.asin.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.asin.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.asin.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.asin.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.asin.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.asin.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.asin.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.asin.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.asin.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.asin.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.asin.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.asin.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.asin.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.asin.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.asin.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.asin.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.asin.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.asin.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.asin.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.asin.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.asin.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.asin.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.asin.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.asin.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.asin.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.asin.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.asin.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.asin.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.asin.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.asin.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.asin.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.asin.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.asin.f32(float)
declare <4 x float> @llvm.fpbuiltin.asin.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.asin.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.asin.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.asin.f64(double)
declare <2 x double> @llvm.fpbuiltin.asin.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.asin.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.asin.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.asin.f16(half)
declare <4 x half> @llvm.fpbuiltin.asin.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.asin.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.asin.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.asin.v32f16(<32 x half>)

; CHECK-LABEL: @svml_asinh
; CHECK: call float @__svml_asinhf1_ha(
; CHECK: call float @__svml_asinhf1(
; CHECK: call float @__svml_asinhf1_ep(
; CHECK: call <4 x float> @__svml_asinhf4_ha(
; CHECK: call <4 x float> @__svml_asinhf4(
; CHECK: call <4 x float> @__svml_asinhf4_ep(
; CHECK: call <8 x float> @__svml_asinhf8_ha(
; CHECK: call <8 x float> @__svml_asinhf8(
; CHECK: call <8 x float> @__svml_asinhf8_ep(
; CHECK: call <16 x float> @__svml_asinhf16_ha(
; CHECK: call <16 x float> @__svml_asinhf16(
; CHECK: call <16 x float> @__svml_asinhf16_ep(
; CHECK: call double @__svml_asinh1_ha(
; CHECK: call double @__svml_asinh1(
; CHECK: call double @__svml_asinh1_ep(
; CHECK: call <2 x double> @__svml_asinh2_ha(
; CHECK: call <2 x double> @__svml_asinh2(
; CHECK: call <2 x double> @__svml_asinh2_ep(
; CHECK: call <4 x double> @__svml_asinh4_ha(
; CHECK: call <4 x double> @__svml_asinh4(
; CHECK: call <4 x double> @__svml_asinh4_ep(
; CHECK: call <8 x double> @__svml_asinh8_ha(
; CHECK: call <8 x double> @__svml_asinh8(
; CHECK: call <8 x double> @__svml_asinh8_ep(
; CHECK: call half @__svml_asinhs1_ha(
; CHECK: call half @__svml_asinhs1(
; CHECK: call half @__svml_asinhs1_ep(
; CHECK: call <4 x half> @__svml_asinhs4_ha(
; CHECK: call <4 x half> @__svml_asinhs4(
; CHECK: call <4 x half> @__svml_asinhs4_ep(
; CHECK: call <8 x half> @__svml_asinhs8_ha(
; CHECK: call <8 x half> @__svml_asinhs8(
; CHECK: call <8 x half> @__svml_asinhs8_ep(
; CHECK: call <16 x half> @__svml_asinhs16_ha(
; CHECK: call <16 x half> @__svml_asinhs16(
; CHECK: call <16 x half> @__svml_asinhs16_ep(
; CHECK: call <32 x half> @__svml_asinhs32_ha(
; CHECK: call <32 x half> @__svml_asinhs32(
; CHECK: call <32 x half> @__svml_asinhs32_ep(
define void @svml_asinh(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                        double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                        half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.asinh.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.asinh.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.asinh.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.asinh.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.asinh.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.asinh.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.asinh.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.asinh.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.asinh.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.asinh.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.asinh.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.asinh.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.asinh.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.asinh.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.asinh.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.asinh.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.asinh.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.asinh.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.asinh.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.asinh.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.asinh.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.asinh.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.asinh.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.asinh.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.asinh.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.asinh.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.asinh.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.asinh.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.asinh.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.asinh.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.asinh.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.asinh.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.asinh.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.asinh.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.asinh.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.asinh.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.asinh.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.asinh.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.asinh.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.asinh.f32(float)
declare <4 x float> @llvm.fpbuiltin.asinh.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.asinh.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.asinh.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.asinh.f64(double)
declare <2 x double> @llvm.fpbuiltin.asinh.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.asinh.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.asinh.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.asinh.f16(half)
declare <4 x half> @llvm.fpbuiltin.asinh.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.asinh.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.asinh.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.asinh.v32f16(<32 x half>)

; CHECK-LABEL: @svml_atan
; CHECK: call float @__svml_atanf1_ha(
; CHECK: call float @__svml_atanf1(
; CHECK: call float @__svml_atanf1_ep(
; CHECK: call <4 x float> @__svml_atanf4_ha(
; CHECK: call <4 x float> @__svml_atanf4(
; CHECK: call <4 x float> @__svml_atanf4_ep(
; CHECK: call <8 x float> @__svml_atanf8_ha(
; CHECK: call <8 x float> @__svml_atanf8(
; CHECK: call <8 x float> @__svml_atanf8_ep(
; CHECK: call <16 x float> @__svml_atanf16_ha(
; CHECK: call <16 x float> @__svml_atanf16(
; CHECK: call <16 x float> @__svml_atanf16_ep(
; CHECK: call double @__svml_atan1_ha(
; CHECK: call double @__svml_atan1(
; CHECK: call double @__svml_atan1_ep(
; CHECK: call <2 x double> @__svml_atan2_ha(
; CHECK: call <2 x double> @__svml_atan2(
; CHECK: call <2 x double> @__svml_atan2_ep(
; CHECK: call <4 x double> @__svml_atan4_ha(
; CHECK: call <4 x double> @__svml_atan4(
; CHECK: call <4 x double> @__svml_atan4_ep(
; CHECK: call <8 x double> @__svml_atan8_ha(
; CHECK: call <8 x double> @__svml_atan8(
; CHECK: call <8 x double> @__svml_atan8_ep(
; CHECK: call half @__svml_atans1_ha(
; CHECK: call half @__svml_atans1(
; CHECK: call half @__svml_atans1_ep(
; CHECK: call <4 x half> @__svml_atans4_ha(
; CHECK: call <4 x half> @__svml_atans4(
; CHECK: call <4 x half> @__svml_atans4_ep(
; CHECK: call <8 x half> @__svml_atans8_ha(
; CHECK: call <8 x half> @__svml_atans8(
; CHECK: call <8 x half> @__svml_atans8_ep(
; CHECK: call <16 x half> @__svml_atans16_ha(
; CHECK: call <16 x half> @__svml_atans16(
; CHECK: call <16 x half> @__svml_atans16_ep(
; CHECK: call <32 x half> @__svml_atans32_ha(
; CHECK: call <32 x half> @__svml_atans32(
; CHECK: call <32 x half> @__svml_atans32_ep(
define void @svml_atan(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.atan.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.atan.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.atan.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.atan.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.atan.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.atan.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.atan.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.atan.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.atan.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.atan.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.atan.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.atan.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.atan.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.atan.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.atan.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.atan.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.atan.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.atan.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.atan.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.atan.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.atan.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.atan.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.atan.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.atan.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.atan.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.atan.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.atan.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.atan.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.atan.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.atan.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.atan.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.atan.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.atan.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.atan.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.atan.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.atan.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.atan.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.atan.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.atan.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.atan.f32(float)
declare <4 x float> @llvm.fpbuiltin.atan.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.atan.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.atan.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.atan.f64(double)
declare <2 x double> @llvm.fpbuiltin.atan.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.atan.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.atan.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.atan.f16(half)
declare <4 x half> @llvm.fpbuiltin.atan.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.atan.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.atan.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.atan.v32f16(<32 x half>)

; CHECK-LABEL: @svml_atanh
; CHECK: call float @__svml_atanhf1_ha(
; CHECK: call float @__svml_atanhf1(
; CHECK: call float @__svml_atanhf1_ep(
; CHECK: call <4 x float> @__svml_atanhf4_ha(
; CHECK: call <4 x float> @__svml_atanhf4(
; CHECK: call <4 x float> @__svml_atanhf4_ep(
; CHECK: call <8 x float> @__svml_atanhf8_ha(
; CHECK: call <8 x float> @__svml_atanhf8(
; CHECK: call <8 x float> @__svml_atanhf8_ep(
; CHECK: call <16 x float> @__svml_atanhf16_ha(
; CHECK: call <16 x float> @__svml_atanhf16(
; CHECK: call <16 x float> @__svml_atanhf16_ep(
; CHECK: call double @__svml_atanh1_ha(
; CHECK: call double @__svml_atanh1(
; CHECK: call double @__svml_atanh1_ep(
; CHECK: call <2 x double> @__svml_atanh2_ha(
; CHECK: call <2 x double> @__svml_atanh2(
; CHECK: call <2 x double> @__svml_atanh2_ep(
; CHECK: call <4 x double> @__svml_atanh4_ha(
; CHECK: call <4 x double> @__svml_atanh4(
; CHECK: call <4 x double> @__svml_atanh4_ep(
; CHECK: call <8 x double> @__svml_atanh8_ha(
; CHECK: call <8 x double> @__svml_atanh8(
; CHECK: call <8 x double> @__svml_atanh8_ep(
; CHECK: call half @__svml_atanhs1_ha(
; CHECK: call half @__svml_atanhs1(
; CHECK: call half @__svml_atanhs1_ep(
; CHECK: call <4 x half> @__svml_atanhs4_ha(
; CHECK: call <4 x half> @__svml_atanhs4(
; CHECK: call <4 x half> @__svml_atanhs4_ep(
; CHECK: call <8 x half> @__svml_atanhs8_ha(
; CHECK: call <8 x half> @__svml_atanhs8(
; CHECK: call <8 x half> @__svml_atanhs8_ep(
; CHECK: call <16 x half> @__svml_atanhs16_ha(
; CHECK: call <16 x half> @__svml_atanhs16(
; CHECK: call <16 x half> @__svml_atanhs16_ep(
; CHECK: call <32 x half> @__svml_atanhs32_ha(
; CHECK: call <32 x half> @__svml_atanhs32(
; CHECK: call <32 x half> @__svml_atanhs32_ep(
define void @svml_atanh(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                        double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                        half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.atanh.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.atanh.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.atanh.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.atanh.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.atanh.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.atanh.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.atanh.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.atanh.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.atanh.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.atanh.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.atanh.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.atanh.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.atanh.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.atanh.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.atanh.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.atanh.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.atanh.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.atanh.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.atanh.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.atanh.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.atanh.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.atanh.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.atanh.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.atanh.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.atanh.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.atanh.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.atanh.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.atanh.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.atanh.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.atanh.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.atanh.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.atanh.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.atanh.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.atanh.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.atanh.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.atanh.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.atanh.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.atanh.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.atanh.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.atanh.f32(float)
declare <4 x float> @llvm.fpbuiltin.atanh.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.atanh.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.atanh.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.atanh.f64(double)
declare <2 x double> @llvm.fpbuiltin.atanh.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.atanh.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.atanh.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.atanh.f16(half)
declare <4 x half> @llvm.fpbuiltin.atanh.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.atanh.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.atanh.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.atanh.v32f16(<32 x half>)

; CHECK-LABEL: @svml_atan2
; CHECK: call float @__svml_atan2f1_ha(
; CHECK: call float @__svml_atan2f1(
; CHECK: call float @__svml_atan2f1_ep(
; CHECK: call <4 x float> @__svml_atan2f4_ha(
; CHECK: call <4 x float> @__svml_atan2f4(
; CHECK: call <4 x float> @__svml_atan2f4_ep(
; CHECK: call <8 x float> @__svml_atan2f8_ha(
; CHECK: call <8 x float> @__svml_atan2f8(
; CHECK: call <8 x float> @__svml_atan2f8_ep(
; CHECK: call <16 x float> @__svml_atan2f16_ha(
; CHECK: call <16 x float> @__svml_atan2f16(
; CHECK: call <16 x float> @__svml_atan2f16_ep(
; CHECK: call double @__svml_atan21_ha(
; CHECK: call double @__svml_atan21(
; CHECK: call double @__svml_atan21_ep(
; CHECK: call <2 x double> @__svml_atan22_ha(
; CHECK: call <2 x double> @__svml_atan22(
; CHECK: call <2 x double> @__svml_atan22_ep(
; CHECK: call <4 x double> @__svml_atan24_ha(
; CHECK: call <4 x double> @__svml_atan24(
; CHECK: call <4 x double> @__svml_atan24_ep(
; CHECK: call <8 x double> @__svml_atan28_ha(
; CHECK: call <8 x double> @__svml_atan28(
; CHECK: call <8 x double> @__svml_atan28_ep(
; CHECK: call half @__svml_atan2s1_ha(
; CHECK: call half @__svml_atan2s1(
; CHECK: call half @__svml_atan2s1_ep(
; CHECK: call <4 x half> @__svml_atan2s4_ha(
; CHECK: call <4 x half> @__svml_atan2s4(
; CHECK: call <4 x half> @__svml_atan2s4_ep(
; CHECK: call <8 x half> @__svml_atan2s8_ha(
; CHECK: call <8 x half> @__svml_atan2s8(
; CHECK: call <8 x half> @__svml_atan2s8_ep(
; CHECK: call <16 x half> @__svml_atan2s16_ha(
; CHECK: call <16 x half> @__svml_atan2s16(
; CHECK: call <16 x half> @__svml_atan2s16_ep(
; CHECK: call <32 x half> @__svml_atan2s32_ha(
; CHECK: call <32 x half> @__svml_atan2s32(
; CHECK: call <32 x half> @__svml_atan2s32_ep(
define void @svml_atan2(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                        float %f2, <4 x float> %v4f2, <8 x float> %v8f2, <16 x float> %v16f2,
                        double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                        double %d2, <2 x double> %v2d2, <4 x double> %v4d2, <8 x double> %v8d2,
                        half %h1, <4 x half> %v4h1, <8 x half> %v8h1, <16 x half> %v16h1, <32 x half> %v32h1,
                        half %h2, <4 x half> %v4h2, <8 x half> %v8h2, <16 x half> %v16h2, <32 x half> %v32h2) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.atan2.f32(float %f1, float %f2) #0
  %t0_1 = call float @llvm.fpbuiltin.atan2.f32(float %f1, float %f2) #1
  %t0_2 = call float @llvm.fpbuiltin.atan2.f32(float %f1, float %f2) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.atan2.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.atan2.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.atan2.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.atan2.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.atan2.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.atan2.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.atan2.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.atan2.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.atan2.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #2
  %t4_0 = call double @llvm.fpbuiltin.atan2.f64(double %d1, double %d2) #0
  %t4_1 = call double @llvm.fpbuiltin.atan2.f64(double %d1, double %d2) #1
  %t4_2 = call double @llvm.fpbuiltin.atan2.f64(double %d1, double %d2) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.atan2.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.atan2.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.atan2.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.atan2.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.atan2.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.atan2.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.atan2.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.atan2.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.atan2.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #3
  %t8_0 = call half @llvm.fpbuiltin.atan2.f16(half %h1, half %h2) #0
  %t8_1 = call half @llvm.fpbuiltin.atan2.f16(half %h1, half %h2) #1
  %t8_2 = call half @llvm.fpbuiltin.atan2.f16(half %h1, half %h2) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.atan2.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.atan2.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.atan2.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.atan2.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.atan2.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.atan2.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.atan2.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.atan2.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.atan2.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.atan2.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.atan2.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.atan2.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #4
  ret void
}

declare float @llvm.fpbuiltin.atan2.f32(float, float)
declare <4 x float> @llvm.fpbuiltin.atan2.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.fpbuiltin.atan2.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.fpbuiltin.atan2.v16f32(<16 x float>, <16 x float>)
declare double @llvm.fpbuiltin.atan2.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.atan2.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.fpbuiltin.atan2.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.fpbuiltin.atan2.v8f64(<8 x double>, <8 x double>)
declare half @llvm.fpbuiltin.atan2.f16(half, half)
declare <4 x half> @llvm.fpbuiltin.atan2.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.fpbuiltin.atan2.v8f16(<8 x half>, <8 x half>)
declare <16 x half> @llvm.fpbuiltin.atan2.v16f16(<16 x half>, <16 x half>)
declare <32 x half> @llvm.fpbuiltin.atan2.v32f16(<32 x half>, <32 x half>)

; CHECK-LABEL: @svml_sincos
; CHECK: call void @__svml_sincosf1_ha(
; CHECK: call void @__svml_sincosf1(
; CHECK: call void @__svml_sincosf1_ep(
; CHECK: call void @__svml_sincosf4_ha(
; CHECK: call void @__svml_sincosf4(
; CHECK: call void @__svml_sincosf4_ep(
; CHECK: call void @__svml_sincosf8_ha(
; CHECK: call void @__svml_sincosf8(
; CHECK: call void @__svml_sincosf8_ep(
; CHECK: call void @__svml_sincosf16_ha(
; CHECK: call void @__svml_sincosf16(
; CHECK: call void @__svml_sincosf16_ep(
; CHECK: call void @__svml_sincos1_ha(
; CHECK: call void @__svml_sincos1(
; CHECK: call void @__svml_sincos1_ep(
; CHECK: call void @__svml_sincos2_ha(
; CHECK: call void @__svml_sincos2(
; CHECK: call void @__svml_sincos2_ep(
; CHECK: call void @__svml_sincos4_ha(
; CHECK: call void @__svml_sincos4(
; CHECK: call void @__svml_sincos4_ep(
; CHECK: call void @__svml_sincos8_ha(
; CHECK: call void @__svml_sincos8(
; CHECK: call void @__svml_sincos8_ep(
; CHECK: call void @__svml_sincoss1_ha(
; CHECK: call void @__svml_sincoss1(
; CHECK: call void @__svml_sincoss1_ep(
; CHECK: call void @__svml_sincoss4_ha(
; CHECK: call void @__svml_sincoss4(
; CHECK: call void @__svml_sincoss4_ep(
; CHECK: call void @__svml_sincoss8_ha(
; CHECK: call void @__svml_sincoss8(
; CHECK: call void @__svml_sincoss8_ep(
; CHECK: call void @__svml_sincoss16_ha(
; CHECK: call void @__svml_sincoss16(
; CHECK: call void @__svml_sincoss16_ep(
; CHECK: call void @__svml_sincoss32_ha(
; CHECK: call void @__svml_sincoss32(
; CHECK: call void @__svml_sincoss32_ep(
define void @svml_sincos(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                         double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                         half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h,
                         ptr %sin, ptr %cos) {
entry:
  call void @llvm.fpbuiltin.sincos.f32(float %f, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.f32(float %f, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.f32(float %f, ptr %sin, ptr %cos) #2
  call void @llvm.fpbuiltin.sincos.v4f32(<4 x float> %v4f, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v4f32(<4 x float> %v4f, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v4f32(<4 x float> %v4f, ptr %sin, ptr %cos) #2
  call void @llvm.fpbuiltin.sincos.v8f32(<8 x float> %v8f, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v8f32(<8 x float> %v8f, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v8f32(<8 x float> %v8f, ptr %sin, ptr %cos) #2
  call void @llvm.fpbuiltin.sincos.v16f32(<16 x float> %v16f, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v16f32(<16 x float> %v16f, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v16f32(<16 x float> %v16f, ptr %sin, ptr %cos) #2
  call void @llvm.fpbuiltin.sincos.f64(double %d, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.f64(double %d, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.f64(double %d, ptr %sin, ptr %cos) #3
  call void @llvm.fpbuiltin.sincos.v2f64(<2 x double> %v2d, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v2f64(<2 x double> %v2d, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v2f64(<2 x double> %v2d, ptr %sin, ptr %cos) #3
  call void @llvm.fpbuiltin.sincos.v4f64(<4 x double> %v4d, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v4f64(<4 x double> %v4d, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v4f64(<4 x double> %v4d, ptr %sin, ptr %cos) #3
  call void @llvm.fpbuiltin.sincos.v8f64(<8 x double> %v8d, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v8f64(<8 x double> %v8d, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v8f64(<8 x double> %v8d, ptr %sin, ptr %cos) #3
  call void @llvm.fpbuiltin.sincos.f16(half %h, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.f16(half %h, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.f16(half %h, ptr %sin, ptr %cos) #4
  call void @llvm.fpbuiltin.sincos.v4f16(<4 x half> %v4h, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v4f16(<4 x half> %v4h, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v4f16(<4 x half> %v4h, ptr %sin, ptr %cos) #4
  call void @llvm.fpbuiltin.sincos.v8f16(<8 x half> %v8h, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v8f16(<8 x half> %v8h, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v8f16(<8 x half> %v8h, ptr %sin, ptr %cos) #4
  call void @llvm.fpbuiltin.sincos.v16f16(<16 x half> %v16h, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v16f16(<16 x half> %v16h, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v16f16(<16 x half> %v16h, ptr %sin, ptr %cos) #4
  call void @llvm.fpbuiltin.sincos.v32f16(<32 x half> %v32h, ptr %sin, ptr %cos) #0
  call void @llvm.fpbuiltin.sincos.v32f16(<32 x half> %v32h, ptr %sin, ptr %cos) #1
  call void @llvm.fpbuiltin.sincos.v32f16(<32 x half> %v32h, ptr %sin, ptr %cos) #4
  ret void
}

declare void @llvm.fpbuiltin.sincos.f32(float, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v4f32(<4 x float>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v8f32(<8 x float>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v16f32(<16 x float>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.f64(double, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v2f64(<2 x double>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v4f64(<4 x double>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v8f64(<8 x double>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.f16(half, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v4f16(<4 x half>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v8f16(<8 x half>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v16f16(<16 x half>, ptr, ptr)
declare void @llvm.fpbuiltin.sincos.v32f16(<32 x half>, ptr, ptr)

; CHECK-LABEL: @svml_erf
; CHECK: call float @__svml_erff1_ha(
; CHECK: call float @__svml_erff1(
; CHECK: call float @__svml_erff1_ep(
; CHECK: call <4 x float> @__svml_erff4_ha(
; CHECK: call <4 x float> @__svml_erff4(
; CHECK: call <4 x float> @__svml_erff4_ep(
; CHECK: call <8 x float> @__svml_erff8_ha(
; CHECK: call <8 x float> @__svml_erff8(
; CHECK: call <8 x float> @__svml_erff8_ep(
; CHECK: call <16 x float> @__svml_erff16_ha(
; CHECK: call <16 x float> @__svml_erff16(
; CHECK: call <16 x float> @__svml_erff16_ep(
; CHECK: call double @__svml_erf1_ha(
; CHECK: call double @__svml_erf1(
; CHECK: call double @__svml_erf1_ep(
; CHECK: call <2 x double> @__svml_erf2_ha(
; CHECK: call <2 x double> @__svml_erf2(
; CHECK: call <2 x double> @__svml_erf2_ep(
; CHECK: call <4 x double> @__svml_erf4_ha(
; CHECK: call <4 x double> @__svml_erf4(
; CHECK: call <4 x double> @__svml_erf4_ep(
; CHECK: call <8 x double> @__svml_erf8_ha(
; CHECK: call <8 x double> @__svml_erf8(
; CHECK: call <8 x double> @__svml_erf8_ep(
; CHECK: call half @__svml_erfs1_ha(
; CHECK: call half @__svml_erfs1(
; CHECK: call half @__svml_erfs1_ep(
; CHECK: call <4 x half> @__svml_erfs4_ha(
; CHECK: call <4 x half> @__svml_erfs4(
; CHECK: call <4 x half> @__svml_erfs4_ep(
; CHECK: call <8 x half> @__svml_erfs8_ha(
; CHECK: call <8 x half> @__svml_erfs8(
; CHECK: call <8 x half> @__svml_erfs8_ep(
; CHECK: call <16 x half> @__svml_erfs16_ha(
; CHECK: call <16 x half> @__svml_erfs16(
; CHECK: call <16 x half> @__svml_erfs16_ep(
; CHECK: call <32 x half> @__svml_erfs32_ha(
; CHECK: call <32 x half> @__svml_erfs32(
; CHECK: call <32 x half> @__svml_erfs32_ep(
define void @svml_erf(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                      double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                      half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.erf.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.erf.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.erf.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.erf.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.erf.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.erf.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.erf.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.erf.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.erf.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.erf.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.erf.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.erf.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.erf.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.erf.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.erf.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.erf.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.erf.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.erf.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.erf.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.erf.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.erf.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.erf.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.erf.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.erf.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.erf.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.erf.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.erf.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.erf.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.erf.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.erf.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.erf.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.erf.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.erf.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.erf.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.erf.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.erf.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.erf.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.erf.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.erf.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.erf.f32(float)
declare <4 x float> @llvm.fpbuiltin.erf.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.erf.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.erf.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.erf.f64(double)
declare <2 x double> @llvm.fpbuiltin.erf.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.erf.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.erf.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.erf.f16(half)
declare <4 x half> @llvm.fpbuiltin.erf.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.erf.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.erf.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.erf.v32f16(<32 x half>)

; CHECK-LABEL: @svml_erfc
; CHECK: call float @__svml_erfcf1_ha(
; CHECK: call float @__svml_erfcf1(
; CHECK: call float @__svml_erfcf1_ep(
; CHECK: call <4 x float> @__svml_erfcf4_ha(
; CHECK: call <4 x float> @__svml_erfcf4(
; CHECK: call <4 x float> @__svml_erfcf4_ep(
; CHECK: call <8 x float> @__svml_erfcf8_ha(
; CHECK: call <8 x float> @__svml_erfcf8(
; CHECK: call <8 x float> @__svml_erfcf8_ep(
; CHECK: call <16 x float> @__svml_erfcf16_ha(
; CHECK: call <16 x float> @__svml_erfcf16(
; CHECK: call <16 x float> @__svml_erfcf16_ep(
; CHECK: call double @__svml_erfc1_ha(
; CHECK: call double @__svml_erfc1(
; CHECK: call double @__svml_erfc1_ep(
; CHECK: call <2 x double> @__svml_erfc2_ha(
; CHECK: call <2 x double> @__svml_erfc2(
; CHECK: call <2 x double> @__svml_erfc2_ep(
; CHECK: call <4 x double> @__svml_erfc4_ha(
; CHECK: call <4 x double> @__svml_erfc4(
; CHECK: call <4 x double> @__svml_erfc4_ep(
; CHECK: call <8 x double> @__svml_erfc8_ha(
; CHECK: call <8 x double> @__svml_erfc8(
; CHECK: call <8 x double> @__svml_erfc8_ep(
; CHECK: call half @__svml_erfcs1_ha(
; CHECK: call half @__svml_erfcs1(
; CHECK: call half @__svml_erfcs1_ep(
; CHECK: call <4 x half> @__svml_erfcs4_ha(
; CHECK: call <4 x half> @__svml_erfcs4(
; CHECK: call <4 x half> @__svml_erfcs4_ep(
; CHECK: call <8 x half> @__svml_erfcs8_ha(
; CHECK: call <8 x half> @__svml_erfcs8(
; CHECK: call <8 x half> @__svml_erfcs8_ep(
; CHECK: call <16 x half> @__svml_erfcs16_ha(
; CHECK: call <16 x half> @__svml_erfcs16(
; CHECK: call <16 x half> @__svml_erfcs16_ep(
; CHECK: call <32 x half> @__svml_erfcs32_ha(
; CHECK: call <32 x half> @__svml_erfcs32(
; CHECK: call <32 x half> @__svml_erfcs32_ep(
define void @svml_erfc(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.erfc.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.erfc.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.erfc.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.erfc.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.erfc.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.erfc.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.erfc.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.erfc.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.erfc.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.erfc.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.erfc.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.erfc.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.erfc.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.erfc.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.erfc.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.erfc.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.erfc.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.erfc.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.erfc.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.erfc.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.erfc.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.erfc.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.erfc.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.erfc.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.erfc.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.erfc.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.erfc.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.erfc.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.erfc.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.erfc.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.erfc.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.erfc.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.erfc.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.erfc.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.erfc.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.erfc.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.erfc.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.erfc.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.erfc.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.erfc.f32(float)
declare <4 x float> @llvm.fpbuiltin.erfc.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.erfc.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.erfc.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.erfc.f64(double)
declare <2 x double> @llvm.fpbuiltin.erfc.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.erfc.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.erfc.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.erfc.f16(half)
declare <4 x half> @llvm.fpbuiltin.erfc.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.erfc.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.erfc.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.erfc.v32f16(<32 x half>)

; CHECK-LABEL: @svml_exp
; CHECK: call float @__svml_expf1_ha(
; CHECK: call float @__svml_expf1(
; CHECK: call float @__svml_expf1_ep(
; CHECK: call <4 x float> @__svml_expf4_ha(
; CHECK: call <4 x float> @__svml_expf4(
; CHECK: call <4 x float> @__svml_expf4_ep(
; CHECK: call <8 x float> @__svml_expf8_ha(
; CHECK: call <8 x float> @__svml_expf8(
; CHECK: call <8 x float> @__svml_expf8_ep(
; CHECK: call <16 x float> @__svml_expf16_ha(
; CHECK: call <16 x float> @__svml_expf16(
; CHECK: call <16 x float> @__svml_expf16_ep(
; CHECK: call double @__svml_exp1_ha(
; CHECK: call double @__svml_exp1(
; CHECK: call double @__svml_exp1_ep(
; CHECK: call <2 x double> @__svml_exp2_ha(
; CHECK: call <2 x double> @__svml_exp2(
; CHECK: call <2 x double> @__svml_exp2_ep(
; CHECK: call <4 x double> @__svml_exp4_ha(
; CHECK: call <4 x double> @__svml_exp4(
; CHECK: call <4 x double> @__svml_exp4_ep(
; CHECK: call <8 x double> @__svml_exp8_ha(
; CHECK: call <8 x double> @__svml_exp8(
; CHECK: call <8 x double> @__svml_exp8_ep(
; CHECK: call half @__svml_exps1_ha(
; CHECK: call half @__svml_exps1(
; CHECK: call half @__svml_exps1_ep(
; CHECK: call <4 x half> @__svml_exps4_ha(
; CHECK: call <4 x half> @__svml_exps4(
; CHECK: call <4 x half> @__svml_exps4_ep(
; CHECK: call <8 x half> @__svml_exps8_ha(
; CHECK: call <8 x half> @__svml_exps8(
; CHECK: call <8 x half> @__svml_exps8_ep(
; CHECK: call <16 x half> @__svml_exps16_ha(
; CHECK: call <16 x half> @__svml_exps16(
; CHECK: call <16 x half> @__svml_exps16_ep(
; CHECK: call <32 x half> @__svml_exps32_ha(
; CHECK: call <32 x half> @__svml_exps32(
; CHECK: call <32 x half> @__svml_exps32_ep(
define void @svml_exp(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                      double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                      half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.exp.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.exp.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.exp.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.exp.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.exp.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.exp.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.exp.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.exp.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.exp.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.exp.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.exp.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.exp.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.exp.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.exp.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.exp.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.exp.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.exp.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.exp.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.exp.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.exp.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.exp.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.exp.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.exp.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.exp.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.exp.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.exp.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.exp.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.exp.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.exp.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.exp.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.exp.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.exp.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.exp.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.exp.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.exp.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.exp.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.exp.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.exp.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.exp.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.exp.f32(float)
declare <4 x float> @llvm.fpbuiltin.exp.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.exp.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.exp.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.exp.f64(double)
declare <2 x double> @llvm.fpbuiltin.exp.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.exp.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.exp.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.exp.f16(half)
declare <4 x half> @llvm.fpbuiltin.exp.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.exp.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.exp.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.exp.v32f16(<32 x half>)

; CHECK-LABEL: @svml_exp2
; CHECK: call float @__svml_exp2f1_ha(
; CHECK: call float @__svml_exp2f1(
; CHECK: call float @__svml_exp2f1_ep(
; CHECK: call <4 x float> @__svml_exp2f4_ha(
; CHECK: call <4 x float> @__svml_exp2f4(
; CHECK: call <4 x float> @__svml_exp2f4_ep(
; CHECK: call <8 x float> @__svml_exp2f8_ha(
; CHECK: call <8 x float> @__svml_exp2f8(
; CHECK: call <8 x float> @__svml_exp2f8_ep(
; CHECK: call <16 x float> @__svml_exp2f16_ha(
; CHECK: call <16 x float> @__svml_exp2f16(
; CHECK: call <16 x float> @__svml_exp2f16_ep(
; CHECK: call double @__svml_exp21_ha(
; CHECK: call double @__svml_exp21(
; CHECK: call double @__svml_exp21_ep(
; CHECK: call <2 x double> @__svml_exp22_ha(
; CHECK: call <2 x double> @__svml_exp22(
; CHECK: call <2 x double> @__svml_exp22_ep(
; CHECK: call <4 x double> @__svml_exp24_ha(
; CHECK: call <4 x double> @__svml_exp24(
; CHECK: call <4 x double> @__svml_exp24_ep(
; CHECK: call <8 x double> @__svml_exp28_ha(
; CHECK: call <8 x double> @__svml_exp28(
; CHECK: call <8 x double> @__svml_exp28_ep(
; CHECK: call half @__svml_exp2s1_ha(
; CHECK: call half @__svml_exp2s1(
; CHECK: call half @__svml_exp2s1_ep(
; CHECK: call <4 x half> @__svml_exp2s4_ha(
; CHECK: call <4 x half> @__svml_exp2s4(
; CHECK: call <4 x half> @__svml_exp2s4_ep(
; CHECK: call <8 x half> @__svml_exp2s8_ha(
; CHECK: call <8 x half> @__svml_exp2s8(
; CHECK: call <8 x half> @__svml_exp2s8_ep(
; CHECK: call <16 x half> @__svml_exp2s16_ha(
; CHECK: call <16 x half> @__svml_exp2s16(
; CHECK: call <16 x half> @__svml_exp2s16_ep(
; CHECK: call <32 x half> @__svml_exp2s32_ha(
; CHECK: call <32 x half> @__svml_exp2s32(
; CHECK: call <32 x half> @__svml_exp2s32_ep(
define void @svml_exp2(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.exp2.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.exp2.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.exp2.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.exp2.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.exp2.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.exp2.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.exp2.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.exp2.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.exp2.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.exp2.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.exp2.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.exp2.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.exp2.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.exp2.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.exp2.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.exp2.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.exp2.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.exp2.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.exp2.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.exp2.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.exp2.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.exp2.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.exp2.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.exp2.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.exp2.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.exp2.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.exp2.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.exp2.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.exp2.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.exp2.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.exp2.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.exp2.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.exp2.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.exp2.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.exp2.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.exp2.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.exp2.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.exp2.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.exp2.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.exp2.f32(float)
declare <4 x float> @llvm.fpbuiltin.exp2.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.exp2.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.exp2.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.exp2.f64(double)
declare <2 x double> @llvm.fpbuiltin.exp2.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.exp2.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.exp2.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.exp2.f16(half)
declare <4 x half> @llvm.fpbuiltin.exp2.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.exp2.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.exp2.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.exp2.v32f16(<32 x half>)

; CHECK-LABEL: @svml_exp10
; CHECK: call float @__svml_exp10f1_ha(
; CHECK: call float @__svml_exp10f1(
; CHECK: call float @__svml_exp10f1_ep(
; CHECK: call <4 x float> @__svml_exp10f4_ha(
; CHECK: call <4 x float> @__svml_exp10f4(
; CHECK: call <4 x float> @__svml_exp10f4_ep(
; CHECK: call <8 x float> @__svml_exp10f8_ha(
; CHECK: call <8 x float> @__svml_exp10f8(
; CHECK: call <8 x float> @__svml_exp10f8_ep(
; CHECK: call <16 x float> @__svml_exp10f16_ha(
; CHECK: call <16 x float> @__svml_exp10f16(
; CHECK: call <16 x float> @__svml_exp10f16_ep(
; CHECK: call double @__svml_exp101_ha(
; CHECK: call double @__svml_exp101(
; CHECK: call double @__svml_exp101_ep(
; CHECK: call <2 x double> @__svml_exp102_ha(
; CHECK: call <2 x double> @__svml_exp102(
; CHECK: call <2 x double> @__svml_exp102_ep(
; CHECK: call <4 x double> @__svml_exp104_ha(
; CHECK: call <4 x double> @__svml_exp104(
; CHECK: call <4 x double> @__svml_exp104_ep(
; CHECK: call <8 x double> @__svml_exp108_ha(
; CHECK: call <8 x double> @__svml_exp108(
; CHECK: call <8 x double> @__svml_exp108_ep(
; CHECK: call half @__svml_exp10s1_ha(
; CHECK: call half @__svml_exp10s1(
; CHECK: call half @__svml_exp10s1_ep(
; CHECK: call <4 x half> @__svml_exp10s4_ha(
; CHECK: call <4 x half> @__svml_exp10s4(
; CHECK: call <4 x half> @__svml_exp10s4_ep(
; CHECK: call <8 x half> @__svml_exp10s8_ha(
; CHECK: call <8 x half> @__svml_exp10s8(
; CHECK: call <8 x half> @__svml_exp10s8_ep(
; CHECK: call <16 x half> @__svml_exp10s16_ha(
; CHECK: call <16 x half> @__svml_exp10s16(
; CHECK: call <16 x half> @__svml_exp10s16_ep(
; CHECK: call <32 x half> @__svml_exp10s32_ha(
; CHECK: call <32 x half> @__svml_exp10s32(
; CHECK: call <32 x half> @__svml_exp10s32_ep(
define void @svml_exp10(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                        double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                        half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.exp10.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.exp10.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.exp10.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.exp10.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.exp10.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.exp10.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.exp10.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.exp10.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.exp10.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.exp10.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.exp10.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.exp10.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.exp10.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.exp10.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.exp10.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.exp10.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.exp10.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.exp10.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.exp10.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.exp10.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.exp10.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.exp10.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.exp10.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.exp10.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.exp10.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.exp10.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.exp10.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.exp10.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.exp10.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.exp10.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.exp10.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.exp10.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.exp10.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.exp10.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.exp10.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.exp10.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.exp10.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.exp10.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.exp10.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.exp10.f32(float)
declare <4 x float> @llvm.fpbuiltin.exp10.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.exp10.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.exp10.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.exp10.f64(double)
declare <2 x double> @llvm.fpbuiltin.exp10.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.exp10.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.exp10.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.exp10.f16(half)
declare <4 x half> @llvm.fpbuiltin.exp10.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.exp10.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.exp10.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.exp10.v32f16(<32 x half>)

; CHECK-LABEL: @svml_expm1
; CHECK: call float @__svml_expm1f1_ha(
; CHECK: call float @__svml_expm1f1(
; CHECK: call float @__svml_expm1f1_ep(
; CHECK: call <4 x float> @__svml_expm1f4_ha(
; CHECK: call <4 x float> @__svml_expm1f4(
; CHECK: call <4 x float> @__svml_expm1f4_ep(
; CHECK: call <8 x float> @__svml_expm1f8_ha(
; CHECK: call <8 x float> @__svml_expm1f8(
; CHECK: call <8 x float> @__svml_expm1f8_ep(
; CHECK: call <16 x float> @__svml_expm1f16_ha(
; CHECK: call <16 x float> @__svml_expm1f16(
; CHECK: call <16 x float> @__svml_expm1f16_ep(
; CHECK: call double @__svml_expm11_ha(
; CHECK: call double @__svml_expm11(
; CHECK: call double @__svml_expm11_ep(
; CHECK: call <2 x double> @__svml_expm12_ha(
; CHECK: call <2 x double> @__svml_expm12(
; CHECK: call <2 x double> @__svml_expm12_ep(
; CHECK: call <4 x double> @__svml_expm14_ha(
; CHECK: call <4 x double> @__svml_expm14(
; CHECK: call <4 x double> @__svml_expm14_ep(
; CHECK: call <8 x double> @__svml_expm18_ha(
; CHECK: call <8 x double> @__svml_expm18(
; CHECK: call <8 x double> @__svml_expm18_ep(
; CHECK: call half @__svml_expm1s1_ha(
; CHECK: call half @__svml_expm1s1(
; CHECK: call half @__svml_expm1s1_ep(
; CHECK: call <4 x half> @__svml_expm1s4_ha(
; CHECK: call <4 x half> @__svml_expm1s4(
; CHECK: call <4 x half> @__svml_expm1s4_ep(
; CHECK: call <8 x half> @__svml_expm1s8_ha(
; CHECK: call <8 x half> @__svml_expm1s8(
; CHECK: call <8 x half> @__svml_expm1s8_ep(
; CHECK: call <16 x half> @__svml_expm1s16_ha(
; CHECK: call <16 x half> @__svml_expm1s16(
; CHECK: call <16 x half> @__svml_expm1s16_ep(
; CHECK: call <32 x half> @__svml_expm1s32_ha(
; CHECK: call <32 x half> @__svml_expm1s32(
; CHECK: call <32 x half> @__svml_expm1s32_ep(
define void @svml_expm1(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                        double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                        half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.expm1.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.expm1.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.expm1.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.expm1.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.expm1.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.expm1.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.expm1.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.expm1.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.expm1.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.expm1.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.expm1.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.expm1.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.expm1.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.expm1.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.expm1.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.expm1.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.expm1.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.expm1.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.expm1.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.expm1.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.expm1.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.expm1.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.expm1.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.expm1.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.expm1.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.expm1.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.expm1.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.expm1.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.expm1.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.expm1.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.expm1.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.expm1.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.expm1.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.expm1.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.expm1.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.expm1.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.expm1.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.expm1.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.expm1.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.expm1.f32(float)
declare <4 x float> @llvm.fpbuiltin.expm1.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.expm1.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.expm1.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.expm1.f64(double)
declare <2 x double> @llvm.fpbuiltin.expm1.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.expm1.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.expm1.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.expm1.f16(half)
declare <4 x half> @llvm.fpbuiltin.expm1.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.expm1.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.expm1.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.expm1.v32f16(<32 x half>)

; CHECK-LABEL: @svml_hypot
; CHECK: call float @__svml_hypotf1_ha(
; CHECK: call float @__svml_hypotf1(
; CHECK: call float @__svml_hypotf1_ep(
; CHECK: call <4 x float> @__svml_hypotf4_ha(
; CHECK: call <4 x float> @__svml_hypotf4(
; CHECK: call <4 x float> @__svml_hypotf4_ep(
; CHECK: call <8 x float> @__svml_hypotf8_ha(
; CHECK: call <8 x float> @__svml_hypotf8(
; CHECK: call <8 x float> @__svml_hypotf8_ep(
; CHECK: call <16 x float> @__svml_hypotf16_ha(
; CHECK: call <16 x float> @__svml_hypotf16(
; CHECK: call <16 x float> @__svml_hypotf16_ep(
; CHECK: call double @__svml_hypot1_ha(
; CHECK: call double @__svml_hypot1(
; CHECK: call double @__svml_hypot1_ep(
; CHECK: call <2 x double> @__svml_hypot2_ha(
; CHECK: call <2 x double> @__svml_hypot2(
; CHECK: call <2 x double> @__svml_hypot2_ep(
; CHECK: call <4 x double> @__svml_hypot4_ha(
; CHECK: call <4 x double> @__svml_hypot4(
; CHECK: call <4 x double> @__svml_hypot4_ep(
; CHECK: call <8 x double> @__svml_hypot8_ha(
; CHECK: call <8 x double> @__svml_hypot8(
; CHECK: call <8 x double> @__svml_hypot8_ep(
; CHECK: call half @__svml_hypots1_ha(
; CHECK: call half @__svml_hypots1(
; CHECK: call half @__svml_hypots1_ep(
; CHECK: call <4 x half> @__svml_hypots4_ha(
; CHECK: call <4 x half> @__svml_hypots4(
; CHECK: call <4 x half> @__svml_hypots4_ep(
; CHECK: call <8 x half> @__svml_hypots8_ha(
; CHECK: call <8 x half> @__svml_hypots8(
; CHECK: call <8 x half> @__svml_hypots8_ep(
; CHECK: call <16 x half> @__svml_hypots16_ha(
; CHECK: call <16 x half> @__svml_hypots16(
; CHECK: call <16 x half> @__svml_hypots16_ep(
; CHECK: call <32 x half> @__svml_hypots32_ha(
; CHECK: call <32 x half> @__svml_hypots32(
; CHECK: call <32 x half> @__svml_hypots32_ep(
define void @svml_hypot(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                        float %f2, <4 x float> %v4f2, <8 x float> %v8f2, <16 x float> %v16f2,
                        double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                        double %d2, <2 x double> %v2d2, <4 x double> %v4d2, <8 x double> %v8d2,
                        half %h1, <4 x half> %v4h1, <8 x half> %v8h1, <16 x half> %v16h1, <32 x half> %v32h1,
                        half %h2, <4 x half> %v4h2, <8 x half> %v8h2, <16 x half> %v16h2, <32 x half> %v32h2) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.hypot.f32(float %f1, float %f2) #0
  %t0_1 = call float @llvm.fpbuiltin.hypot.f32(float %f1, float %f2) #1
  %t0_2 = call float @llvm.fpbuiltin.hypot.f32(float %f1, float %f2) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.hypot.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.hypot.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.hypot.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.hypot.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.hypot.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.hypot.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.hypot.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.hypot.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.hypot.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #2
  %t4_0 = call double @llvm.fpbuiltin.hypot.f64(double %d1, double %d2) #0
  %t4_1 = call double @llvm.fpbuiltin.hypot.f64(double %d1, double %d2) #1
  %t4_2 = call double @llvm.fpbuiltin.hypot.f64(double %d1, double %d2) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.hypot.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.hypot.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.hypot.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.hypot.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.hypot.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.hypot.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.hypot.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.hypot.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.hypot.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #3
  %t8_0 = call half @llvm.fpbuiltin.hypot.f16(half %h1, half %h2) #0
  %t8_1 = call half @llvm.fpbuiltin.hypot.f16(half %h1, half %h2) #1
  %t8_2 = call half @llvm.fpbuiltin.hypot.f16(half %h1, half %h2) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.hypot.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.hypot.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.hypot.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.hypot.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.hypot.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.hypot.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.hypot.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.hypot.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.hypot.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.hypot.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.hypot.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.hypot.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #4
  ret void
}

declare float @llvm.fpbuiltin.hypot.f32(float, float)
declare <4 x float> @llvm.fpbuiltin.hypot.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.fpbuiltin.hypot.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.fpbuiltin.hypot.v16f32(<16 x float>, <16 x float>)
declare double @llvm.fpbuiltin.hypot.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.hypot.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.fpbuiltin.hypot.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.fpbuiltin.hypot.v8f64(<8 x double>, <8 x double>)
declare half @llvm.fpbuiltin.hypot.f16(half, half)
declare <4 x half> @llvm.fpbuiltin.hypot.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.fpbuiltin.hypot.v8f16(<8 x half>, <8 x half>)
declare <16 x half> @llvm.fpbuiltin.hypot.v16f16(<16 x half>, <16 x half>)
declare <32 x half> @llvm.fpbuiltin.hypot.v32f16(<32 x half>, <32 x half>)

; CHECK-LABEL: @svml_ldexp
; CHECK: call float @__svml_ldexpf1(
; CHECK: call <4 x float> @__svml_ldexpf4(
; CHECK: call <8 x float> @__svml_ldexpf8(
; CHECK: call <16 x float> @__svml_ldexpf16(
; CHECK: call double @__svml_ldexp1(
; CHECK: call <2 x double> @__svml_ldexp2(
; CHECK: call <4 x double> @__svml_ldexp4(
; CHECK: call <8 x double> @__svml_ldexp8(
define void @svml_ldexp(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                        i32 %f2, <4 x i32> %v4f2, <8 x i32> %v8f2, <16 x i32> %v16f2,
                        double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                        i32 %d2, <2 x i32> %v2d2, <4 x i32> %v4d2, <8 x i32> %v8d2) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.ldexp.f32(float %f1, i32 %f2) #0
  %t1_0 = call <4 x float> @llvm.fpbuiltin.ldexp.v4f32(<4 x float> %v4f1, <4 x i32> %v4f2) #0
  %t2_0 = call <8 x float> @llvm.fpbuiltin.ldexp.v8f32(<8 x float> %v8f1, <8 x i32> %v8f2) #0
  %t3_0 = call <16 x float> @llvm.fpbuiltin.ldexp.v16f32(<16 x float> %v16f1, <16 x i32> %v16f2) #0
  %t4_0 = call double @llvm.fpbuiltin.ldexp.f64(double %d1, i32 %d2) #0
  %t5_0 = call <2 x double> @llvm.fpbuiltin.ldexp.v2f64(<2 x double> %v2d1, <2 x i32> %v2d2) #0
  %t6_0 = call <4 x double> @llvm.fpbuiltin.ldexp.v4f64(<4 x double> %v4d1, <4 x i32> %v4d2) #0
  %t7_0 = call <8 x double> @llvm.fpbuiltin.ldexp.v8f64(<8 x double> %v8d1, <8 x i32> %v8d2) #0
  ret void
}

declare float @llvm.fpbuiltin.ldexp.f32(float, i32)
declare <4 x float> @llvm.fpbuiltin.ldexp.v4f32(<4 x float>, <4 x i32>)
declare <8 x float> @llvm.fpbuiltin.ldexp.v8f32(<8 x float>, <8 x i32>)
declare <16 x float> @llvm.fpbuiltin.ldexp.v16f32(<16 x float>, <16 x i32>)
declare double @llvm.fpbuiltin.ldexp.f64(double, i32)
declare <2 x double> @llvm.fpbuiltin.ldexp.v2f64(<2 x double>, <2 x i32>)
declare <4 x double> @llvm.fpbuiltin.ldexp.v4f64(<4 x double>, <4 x i32>)
declare <8 x double> @llvm.fpbuiltin.ldexp.v8f64(<8 x double>, <8 x i32>)

; CHECK-LABEL: @svml_log
; CHECK: call float @__svml_logf1_ha(
; CHECK: call float @__svml_logf1(
; CHECK: call float @__svml_logf1_ep(
; CHECK: call <4 x float> @__svml_logf4_ha(
; CHECK: call <4 x float> @__svml_logf4(
; CHECK: call <4 x float> @__svml_logf4_ep(
; CHECK: call <8 x float> @__svml_logf8_ha(
; CHECK: call <8 x float> @__svml_logf8(
; CHECK: call <8 x float> @__svml_logf8_ep(
; CHECK: call <16 x float> @__svml_logf16_ha(
; CHECK: call <16 x float> @__svml_logf16(
; CHECK: call <16 x float> @__svml_logf16_ep(
; CHECK: call double @__svml_log1_ha(
; CHECK: call double @__svml_log1(
; CHECK: call double @__svml_log1_ep(
; CHECK: call <2 x double> @__svml_log2_ha(
; CHECK: call <2 x double> @__svml_log2(
; CHECK: call <2 x double> @__svml_log2_ep(
; CHECK: call <4 x double> @__svml_log4_ha(
; CHECK: call <4 x double> @__svml_log4(
; CHECK: call <4 x double> @__svml_log4_ep(
; CHECK: call <8 x double> @__svml_log8_ha(
; CHECK: call <8 x double> @__svml_log8(
; CHECK: call <8 x double> @__svml_log8_ep(
; CHECK: call half @__svml_logs1_ha(
; CHECK: call half @__svml_logs1(
; CHECK: call half @__svml_logs1_ep(
; CHECK: call <4 x half> @__svml_logs4_ha(
; CHECK: call <4 x half> @__svml_logs4(
; CHECK: call <4 x half> @__svml_logs4_ep(
; CHECK: call <8 x half> @__svml_logs8_ha(
; CHECK: call <8 x half> @__svml_logs8(
; CHECK: call <8 x half> @__svml_logs8_ep(
; CHECK: call <16 x half> @__svml_logs16_ha(
; CHECK: call <16 x half> @__svml_logs16(
; CHECK: call <16 x half> @__svml_logs16_ep(
; CHECK: call <32 x half> @__svml_logs32_ha(
; CHECK: call <32 x half> @__svml_logs32(
; CHECK: call <32 x half> @__svml_logs32_ep(
define void @svml_log(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                      double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                      half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.log.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.log.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.log.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.log.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.log.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.log.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.log.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.log.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.log.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.log.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.log.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.log.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.log.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.log.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.log.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.log.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.log.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.log.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.log.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.log.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.log.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.log.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.log.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.log.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.log.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.log.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.log.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.log.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.log.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.log.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.log.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.log.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.log.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.log.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.log.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.log.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.log.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.log.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.log.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.log.f32(float)
declare <4 x float> @llvm.fpbuiltin.log.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.log.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.log.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.log.f64(double)
declare <2 x double> @llvm.fpbuiltin.log.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.log.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.log.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.log.f16(half)
declare <4 x half> @llvm.fpbuiltin.log.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.log.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.log.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.log.v32f16(<32 x half>)

; CHECK-LABEL: @svml_log10
; CHECK: call float @__svml_log10f1_ha(
; CHECK: call float @__svml_log10f1(
; CHECK: call float @__svml_log10f1_ep(
; CHECK: call <4 x float> @__svml_log10f4_ha(
; CHECK: call <4 x float> @__svml_log10f4(
; CHECK: call <4 x float> @__svml_log10f4_ep(
; CHECK: call <8 x float> @__svml_log10f8_ha(
; CHECK: call <8 x float> @__svml_log10f8(
; CHECK: call <8 x float> @__svml_log10f8_ep(
; CHECK: call <16 x float> @__svml_log10f16_ha(
; CHECK: call <16 x float> @__svml_log10f16(
; CHECK: call <16 x float> @__svml_log10f16_ep(
; CHECK: call double @__svml_log101_ha(
; CHECK: call double @__svml_log101(
; CHECK: call double @__svml_log101_ep(
; CHECK: call <2 x double> @__svml_log102_ha(
; CHECK: call <2 x double> @__svml_log102(
; CHECK: call <2 x double> @__svml_log102_ep(
; CHECK: call <4 x double> @__svml_log104_ha(
; CHECK: call <4 x double> @__svml_log104(
; CHECK: call <4 x double> @__svml_log104_ep(
; CHECK: call <8 x double> @__svml_log108_ha(
; CHECK: call <8 x double> @__svml_log108(
; CHECK: call <8 x double> @__svml_log108_ep(
; CHECK: call half @__svml_log10s1_ha(
; CHECK: call half @__svml_log10s1(
; CHECK: call half @__svml_log10s1_ep(
; CHECK: call <4 x half> @__svml_log10s4_ha(
; CHECK: call <4 x half> @__svml_log10s4(
; CHECK: call <4 x half> @__svml_log10s4_ep(
; CHECK: call <8 x half> @__svml_log10s8_ha(
; CHECK: call <8 x half> @__svml_log10s8(
; CHECK: call <8 x half> @__svml_log10s8_ep(
; CHECK: call <16 x half> @__svml_log10s16_ha(
; CHECK: call <16 x half> @__svml_log10s16(
; CHECK: call <16 x half> @__svml_log10s16_ep(
; CHECK: call <32 x half> @__svml_log10s32_ha(
; CHECK: call <32 x half> @__svml_log10s32(
; CHECK: call <32 x half> @__svml_log10s32_ep(
define void @svml_log10(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                        double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                        half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.log10.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.log10.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.log10.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.log10.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.log10.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.log10.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.log10.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.log10.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.log10.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.log10.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.log10.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.log10.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.log10.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.log10.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.log10.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.log10.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.log10.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.log10.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.log10.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.log10.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.log10.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.log10.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.log10.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.log10.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.log10.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.log10.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.log10.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.log10.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.log10.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.log10.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.log10.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.log10.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.log10.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.log10.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.log10.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.log10.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.log10.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.log10.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.log10.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.log10.f32(float)
declare <4 x float> @llvm.fpbuiltin.log10.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.log10.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.log10.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.log10.f64(double)
declare <2 x double> @llvm.fpbuiltin.log10.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.log10.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.log10.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.log10.f16(half)
declare <4 x half> @llvm.fpbuiltin.log10.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.log10.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.log10.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.log10.v32f16(<32 x half>)

; CHECK-LABEL: @svml_log1p
; CHECK: call float @__svml_log1pf1_ha(
; CHECK: call float @__svml_log1pf1(
; CHECK: call float @__svml_log1pf1_ep(
; CHECK: call <4 x float> @__svml_log1pf4_ha(
; CHECK: call <4 x float> @__svml_log1pf4(
; CHECK: call <4 x float> @__svml_log1pf4_ep(
; CHECK: call <8 x float> @__svml_log1pf8_ha(
; CHECK: call <8 x float> @__svml_log1pf8(
; CHECK: call <8 x float> @__svml_log1pf8_ep(
; CHECK: call <16 x float> @__svml_log1pf16_ha(
; CHECK: call <16 x float> @__svml_log1pf16(
; CHECK: call <16 x float> @__svml_log1pf16_ep(
; CHECK: call double @__svml_log1p1_ha(
; CHECK: call double @__svml_log1p1(
; CHECK: call double @__svml_log1p1_ep(
; CHECK: call <2 x double> @__svml_log1p2_ha(
; CHECK: call <2 x double> @__svml_log1p2(
; CHECK: call <2 x double> @__svml_log1p2_ep(
; CHECK: call <4 x double> @__svml_log1p4_ha(
; CHECK: call <4 x double> @__svml_log1p4(
; CHECK: call <4 x double> @__svml_log1p4_ep(
; CHECK: call <8 x double> @__svml_log1p8_ha(
; CHECK: call <8 x double> @__svml_log1p8(
; CHECK: call <8 x double> @__svml_log1p8_ep(
; CHECK: call half @__svml_log1ps1_ha(
; CHECK: call half @__svml_log1ps1(
; CHECK: call half @__svml_log1ps1_ep(
; CHECK: call <4 x half> @__svml_log1ps4_ha(
; CHECK: call <4 x half> @__svml_log1ps4(
; CHECK: call <4 x half> @__svml_log1ps4_ep(
; CHECK: call <8 x half> @__svml_log1ps8_ha(
; CHECK: call <8 x half> @__svml_log1ps8(
; CHECK: call <8 x half> @__svml_log1ps8_ep(
; CHECK: call <16 x half> @__svml_log1ps16_ha(
; CHECK: call <16 x half> @__svml_log1ps16(
; CHECK: call <16 x half> @__svml_log1ps16_ep(
; CHECK: call <32 x half> @__svml_log1ps32_ha(
; CHECK: call <32 x half> @__svml_log1ps32(
; CHECK: call <32 x half> @__svml_log1ps32_ep(
define void @svml_log1p(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                        double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                        half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.log1p.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.log1p.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.log1p.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.log1p.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.log1p.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.log1p.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.log1p.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.log1p.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.log1p.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.log1p.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.log1p.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.log1p.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.log1p.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.log1p.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.log1p.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.log1p.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.log1p.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.log1p.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.log1p.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.log1p.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.log1p.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.log1p.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.log1p.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.log1p.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.log1p.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.log1p.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.log1p.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.log1p.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.log1p.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.log1p.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.log1p.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.log1p.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.log1p.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.log1p.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.log1p.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.log1p.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.log1p.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.log1p.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.log1p.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.log1p.f32(float)
declare <4 x float> @llvm.fpbuiltin.log1p.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.log1p.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.log1p.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.log1p.f64(double)
declare <2 x double> @llvm.fpbuiltin.log1p.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.log1p.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.log1p.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.log1p.f16(half)
declare <4 x half> @llvm.fpbuiltin.log1p.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.log1p.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.log1p.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.log1p.v32f16(<32 x half>)

; CHECK-LABEL: @svml_log2
; CHECK: call float @__svml_log2f1_ha(
; CHECK: call float @__svml_log2f1(
; CHECK: call float @__svml_log2f1_ep(
; CHECK: call <4 x float> @__svml_log2f4_ha(
; CHECK: call <4 x float> @__svml_log2f4(
; CHECK: call <4 x float> @__svml_log2f4_ep(
; CHECK: call <8 x float> @__svml_log2f8_ha(
; CHECK: call <8 x float> @__svml_log2f8(
; CHECK: call <8 x float> @__svml_log2f8_ep(
; CHECK: call <16 x float> @__svml_log2f16_ha(
; CHECK: call <16 x float> @__svml_log2f16(
; CHECK: call <16 x float> @__svml_log2f16_ep(
; CHECK: call double @__svml_log21_ha(
; CHECK: call double @__svml_log21(
; CHECK: call double @__svml_log21_ep(
; CHECK: call <2 x double> @__svml_log22_ha(
; CHECK: call <2 x double> @__svml_log22(
; CHECK: call <2 x double> @__svml_log22_ep(
; CHECK: call <4 x double> @__svml_log24_ha(
; CHECK: call <4 x double> @__svml_log24(
; CHECK: call <4 x double> @__svml_log24_ep(
; CHECK: call <8 x double> @__svml_log28_ha(
; CHECK: call <8 x double> @__svml_log28(
; CHECK: call <8 x double> @__svml_log28_ep(
; CHECK: call half @__svml_log2s1_ha(
; CHECK: call half @__svml_log2s1(
; CHECK: call half @__svml_log2s1_ep(
; CHECK: call <4 x half> @__svml_log2s4_ha(
; CHECK: call <4 x half> @__svml_log2s4(
; CHECK: call <4 x half> @__svml_log2s4_ep(
; CHECK: call <8 x half> @__svml_log2s8_ha(
; CHECK: call <8 x half> @__svml_log2s8(
; CHECK: call <8 x half> @__svml_log2s8_ep(
; CHECK: call <16 x half> @__svml_log2s16_ha(
; CHECK: call <16 x half> @__svml_log2s16(
; CHECK: call <16 x half> @__svml_log2s16_ep(
; CHECK: call <32 x half> @__svml_log2s32_ha(
; CHECK: call <32 x half> @__svml_log2s32(
; CHECK: call <32 x half> @__svml_log2s32_ep(
define void @svml_log2(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.log2.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.log2.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.log2.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.log2.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.log2.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.log2.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.log2.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.log2.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.log2.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.log2.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.log2.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.log2.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.log2.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.log2.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.log2.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.log2.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.log2.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.log2.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.log2.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.log2.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.log2.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.log2.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.log2.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.log2.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.log2.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.log2.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.log2.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.log2.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.log2.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.log2.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.log2.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.log2.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.log2.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.log2.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.log2.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.log2.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.log2.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.log2.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.log2.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.log2.f32(float)
declare <4 x float> @llvm.fpbuiltin.log2.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.log2.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.log2.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.log2.f64(double)
declare <2 x double> @llvm.fpbuiltin.log2.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.log2.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.log2.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.log2.f16(half)
declare <4 x half> @llvm.fpbuiltin.log2.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.log2.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.log2.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.log2.v32f16(<32 x half>)

;FIXME add no-opaque-pointer tests.
;FIXME LanRef.rst mentions this builtin's return type is not void
; CHECK-LABEL: @svml_pow
; CHECK: call float @__svml_powf1_ha(
; CHECK: call float @__svml_powf1(
; CHECK: call float @__svml_powf1_ep(
; CHECK: call <4 x float> @__svml_powf4_ha(
; CHECK: call <4 x float> @__svml_powf4(
; CHECK: call <4 x float> @__svml_powf4_ep(
; CHECK: call <8 x float> @__svml_powf8_ha(
; CHECK: call <8 x float> @__svml_powf8(
; CHECK: call <8 x float> @__svml_powf8_ep(
; CHECK: call <16 x float> @__svml_powf16_ha(
; CHECK: call <16 x float> @__svml_powf16(
; CHECK: call <16 x float> @__svml_powf16_ep(
; CHECK: call double @__svml_pow1_ha(
; CHECK: call double @__svml_pow1(
; CHECK: call double @__svml_pow1_ep(
; CHECK: call <2 x double> @__svml_pow2_ha(
; CHECK: call <2 x double> @__svml_pow2(
; CHECK: call <2 x double> @__svml_pow2_ep(
; CHECK: call <4 x double> @__svml_pow4_ha(
; CHECK: call <4 x double> @__svml_pow4(
; CHECK: call <4 x double> @__svml_pow4_ep(
; CHECK: call <8 x double> @__svml_pow8_ha(
; CHECK: call <8 x double> @__svml_pow8(
; CHECK: call <8 x double> @__svml_pow8_ep(
; CHECK: call half @__svml_pows1_ha(
; CHECK: call half @__svml_pows1(
; CHECK: call half @__svml_pows1_ep(
; CHECK: call <4 x half> @__svml_pows4_ha(
; CHECK: call <4 x half> @__svml_pows4(
; CHECK: call <4 x half> @__svml_pows4_ep(
; CHECK: call <8 x half> @__svml_pows8_ha(
; CHECK: call <8 x half> @__svml_pows8(
; CHECK: call <8 x half> @__svml_pows8_ep(
; CHECK: call <16 x half> @__svml_pows16_ha(
; CHECK: call <16 x half> @__svml_pows16(
; CHECK: call <16 x half> @__svml_pows16_ep(
; CHECK: call <32 x half> @__svml_pows32_ha(
; CHECK: call <32 x half> @__svml_pows32(
; CHECK: call <32 x half> @__svml_pows32_ep(
define void @svml_pow(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                        float %f2, <4 x float> %v4f2, <8 x float> %v8f2, <16 x float> %v16f2,
                        double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                        double %d2, <2 x double> %v2d2, <4 x double> %v4d2, <8 x double> %v8d2,
                        half %h1, <4 x half> %v4h1, <8 x half> %v8h1, <16 x half> %v16h1, <32 x half> %v32h1,
                        half %h2, <4 x half> %v4h2, <8 x half> %v8h2, <16 x half> %v16h2, <32 x half> %v32h2) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.pow.f32(float %f1, float %f2) #0
  %t0_1 = call float @llvm.fpbuiltin.pow.f32(float %f1, float %f2) #1
  %t0_2 = call float @llvm.fpbuiltin.pow.f32(float %f1, float %f2) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.pow.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.pow.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.pow.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.pow.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.pow.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.pow.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.pow.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.pow.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.pow.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #2
  %t4_0 = call double @llvm.fpbuiltin.pow.f64(double %d1, double %d2) #0
  %t4_1 = call double @llvm.fpbuiltin.pow.f64(double %d1, double %d2) #1
  %t4_2 = call double @llvm.fpbuiltin.pow.f64(double %d1, double %d2) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.pow.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.pow.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.pow.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.pow.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.pow.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.pow.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.pow.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.pow.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.pow.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #3
  %t8_0 = call half @llvm.fpbuiltin.pow.f16(half %h1, half %h2) #0
  %t8_1 = call half @llvm.fpbuiltin.pow.f16(half %h1, half %h2) #1
  %t8_2 = call half @llvm.fpbuiltin.pow.f16(half %h1, half %h2) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.pow.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.pow.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.pow.v4f16(<4 x half> %v4h1, <4 x half> %v4h2) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.pow.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.pow.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.pow.v8f16(<8 x half> %v8h1, <8 x half> %v8h2) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.pow.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.pow.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.pow.v16f16(<16 x half> %v16h1, <16 x half> %v16h2) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.pow.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.pow.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.pow.v32f16(<32 x half> %v32h1, <32 x half> %v32h2) #4
  ret void
}

declare float @llvm.fpbuiltin.pow.f32(float, float)
declare <4 x float> @llvm.fpbuiltin.pow.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.fpbuiltin.pow.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.fpbuiltin.pow.v16f32(<16 x float>, <16 x float>)
declare double @llvm.fpbuiltin.pow.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.pow.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.fpbuiltin.pow.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.fpbuiltin.pow.v8f64(<8 x double>, <8 x double>)
declare half @llvm.fpbuiltin.pow.f16(half, half)
declare <4 x half> @llvm.fpbuiltin.pow.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.fpbuiltin.pow.v8f16(<8 x half>, <8 x half>)
declare <16 x half> @llvm.fpbuiltin.pow.v16f16(<16 x half>, <16 x half>)
declare <32 x half> @llvm.fpbuiltin.pow.v32f16(<32 x half>, <32 x half>)

; CHECK-LABEL: @svml_sqrt
; CHECK: call float @__svml_sqrtf1_ha(
; CHECK: call float @__svml_sqrtf1(
; CHECK: call float @__svml_sqrtf1_ep(
; CHECK: call <4 x float> @__svml_sqrtf4_ha(
; CHECK: call <4 x float> @__svml_sqrtf4(
; CHECK: call <4 x float> @__svml_sqrtf4_ep(
; CHECK: call <8 x float> @__svml_sqrtf8_ha(
; CHECK: call <8 x float> @__svml_sqrtf8(
; CHECK: call <8 x float> @__svml_sqrtf8_ep(
; CHECK: call <16 x float> @__svml_sqrtf16_ha(
; CHECK: call <16 x float> @__svml_sqrtf16(
; CHECK: call <16 x float> @__svml_sqrtf16_ep(
; CHECK: call double @__svml_sqrt1_ha(
; CHECK: call double @__svml_sqrt1(
; CHECK: call double @__svml_sqrt1_ep(
; CHECK: call <2 x double> @__svml_sqrt2_ha(
; CHECK: call <2 x double> @__svml_sqrt2(
; CHECK: call <2 x double> @__svml_sqrt2_ep(
; CHECK: call <4 x double> @__svml_sqrt4_ha(
; CHECK: call <4 x double> @__svml_sqrt4(
; CHECK: call <4 x double> @__svml_sqrt4_ep(
; CHECK: call <8 x double> @__svml_sqrt8_ha(
; CHECK: call <8 x double> @__svml_sqrt8(
; CHECK: call <8 x double> @__svml_sqrt8_ep(
; CHECK: call half @__svml_sqrts1_ha(
; CHECK: call half @__svml_sqrts1(
; CHECK: call half @__svml_sqrts1_ep(
; CHECK: call <4 x half> @__svml_sqrts4_ha(
; CHECK: call <4 x half> @__svml_sqrts4(
; CHECK: call <4 x half> @__svml_sqrts4_ep(
; CHECK: call <8 x half> @__svml_sqrts8_ha(
; CHECK: call <8 x half> @__svml_sqrts8(
; CHECK: call <8 x half> @__svml_sqrts8_ep(
; CHECK: call <16 x half> @__svml_sqrts16_ha(
; CHECK: call <16 x half> @__svml_sqrts16(
; CHECK: call <16 x half> @__svml_sqrts16_ep(
; CHECK: call <32 x half> @__svml_sqrts32_ha(
; CHECK: call <32 x half> @__svml_sqrts32(
; CHECK: call <32 x half> @__svml_sqrts32_ep(
define void @svml_sqrt(float %f, <4 x float> %v4f, <8 x float> %v8f, <16 x float> %v16f,
                       double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d,
                       half %h, <4 x half> %v4h, <8 x half> %v8h, <16 x half> %v16h, <32 x half> %v32h) {
entry:
  %t0_0 = call float @llvm.fpbuiltin.sqrt.f32(float %f) #0
  %t0_1 = call float @llvm.fpbuiltin.sqrt.f32(float %f) #1
  %t0_2 = call float @llvm.fpbuiltin.sqrt.f32(float %f) #2
  %t1_0 = call <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float> %v4f) #0
  %t1_1 = call <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float> %v4f) #1
  %t1_2 = call <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float> %v4f) #2
  %t2_0 = call <8 x float> @llvm.fpbuiltin.sqrt.v8f32(<8 x float> %v8f) #0
  %t2_1 = call <8 x float> @llvm.fpbuiltin.sqrt.v8f32(<8 x float> %v8f) #1
  %t2_2 = call <8 x float> @llvm.fpbuiltin.sqrt.v8f32(<8 x float> %v8f) #2
  %t3_0 = call <16 x float> @llvm.fpbuiltin.sqrt.v16f32(<16 x float> %v16f) #0
  %t3_1 = call <16 x float> @llvm.fpbuiltin.sqrt.v16f32(<16 x float> %v16f) #1
  %t3_2 = call <16 x float> @llvm.fpbuiltin.sqrt.v16f32(<16 x float> %v16f) #2
  %t4_0 = call double @llvm.fpbuiltin.sqrt.f64(double %d) #0
  %t4_1 = call double @llvm.fpbuiltin.sqrt.f64(double %d) #1
  %t4_2 = call double @llvm.fpbuiltin.sqrt.f64(double %d) #3
  %t5_0 = call <2 x double> @llvm.fpbuiltin.sqrt.v2f64(<2 x double> %v2d) #0
  %t5_1 = call <2 x double> @llvm.fpbuiltin.sqrt.v2f64(<2 x double> %v2d) #1
  %t5_2 = call <2 x double> @llvm.fpbuiltin.sqrt.v2f64(<2 x double> %v2d) #3
  %t6_0 = call <4 x double> @llvm.fpbuiltin.sqrt.v4f64(<4 x double> %v4d) #0
  %t6_1 = call <4 x double> @llvm.fpbuiltin.sqrt.v4f64(<4 x double> %v4d) #1
  %t6_2 = call <4 x double> @llvm.fpbuiltin.sqrt.v4f64(<4 x double> %v4d) #3
  %t7_0 = call <8 x double> @llvm.fpbuiltin.sqrt.v8f64(<8 x double> %v8d) #0
  %t7_1 = call <8 x double> @llvm.fpbuiltin.sqrt.v8f64(<8 x double> %v8d) #1
  %t7_2 = call <8 x double> @llvm.fpbuiltin.sqrt.v8f64(<8 x double> %v8d) #3
  %t8_0 = call half @llvm.fpbuiltin.sqrt.f16(half %h) #0
  %t8_1 = call half @llvm.fpbuiltin.sqrt.f16(half %h) #1
  %t8_2 = call half @llvm.fpbuiltin.sqrt.f16(half %h) #4
  %t9_0 = call <4 x half> @llvm.fpbuiltin.sqrt.v4f16(<4 x half> %v4h) #0
  %t9_1 = call <4 x half> @llvm.fpbuiltin.sqrt.v4f16(<4 x half> %v4h) #1
  %t9_2 = call <4 x half> @llvm.fpbuiltin.sqrt.v4f16(<4 x half> %v4h) #4
  %t10_0 = call <8 x half> @llvm.fpbuiltin.sqrt.v8f16(<8 x half> %v8h) #0
  %t10_1 = call <8 x half> @llvm.fpbuiltin.sqrt.v8f16(<8 x half> %v8h) #1
  %t10_2 = call <8 x half> @llvm.fpbuiltin.sqrt.v8f16(<8 x half> %v8h) #4
  %t11_0 = call <16 x half> @llvm.fpbuiltin.sqrt.v16f16(<16 x half> %v16h) #0
  %t11_1 = call <16 x half> @llvm.fpbuiltin.sqrt.v16f16(<16 x half> %v16h) #1
  %t11_2 = call <16 x half> @llvm.fpbuiltin.sqrt.v16f16(<16 x half> %v16h) #4
  %t12_0 = call <32 x half> @llvm.fpbuiltin.sqrt.v32f16(<32 x half> %v32h) #0
  %t12_1 = call <32 x half> @llvm.fpbuiltin.sqrt.v32f16(<32 x half> %v32h) #1
  %t12_2 = call <32 x half> @llvm.fpbuiltin.sqrt.v32f16(<32 x half> %v32h) #4
  ret void
}

declare float @llvm.fpbuiltin.sqrt.f32(float)
declare <4 x float> @llvm.fpbuiltin.sqrt.v4f32(<4 x float>)
declare <8 x float> @llvm.fpbuiltin.sqrt.v8f32(<8 x float>)
declare <16 x float> @llvm.fpbuiltin.sqrt.v16f32(<16 x float>)
declare double @llvm.fpbuiltin.sqrt.f64(double)
declare <2 x double> @llvm.fpbuiltin.sqrt.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.sqrt.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.sqrt.v8f64(<8 x double>)
declare half @llvm.fpbuiltin.sqrt.f16(half)
declare <4 x half> @llvm.fpbuiltin.sqrt.v4f16(<4 x half>)
declare <8 x half> @llvm.fpbuiltin.sqrt.v8f16(<8 x half>)
declare <16 x half> @llvm.fpbuiltin.sqrt.v16f16(<16 x half>)
declare <32 x half> @llvm.fpbuiltin.sqrt.v32f16(<32 x half>)

attributes #0 = { "fpbuiltin-max-error"="0.6" }
attributes #1 = { "fpbuiltin-max-error"="4.0" }
attributes #2 = { "fpbuiltin-max-error"="4096.0" }
attributes #3 = { "fpbuiltin-max-error"="67108864.0" }
attributes #4 = { "fpbuiltin-max-error"="32" }