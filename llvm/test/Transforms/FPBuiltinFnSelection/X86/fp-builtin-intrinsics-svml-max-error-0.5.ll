; RUN: opt -alt-math-library=svml -fpbuiltin-fn-selection -S < %s | FileCheck %s

; Several functions for "sycl" and "cuda" requires "0.5" accuracy levels,
; Test if these fpbuiltins could be replaced by equivalaent IR operations
; or llvm builtins.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @svml_fadd
; CHECK: %0 = fadd fast float %f1, %f2
; CHECK: %1 = fadd fast <4 x float> %v4f1, %v4f2
; CHECK: %2 = fadd fast <8 x float> %v8f1, %v8f2
; CHECK: %3 = fadd fast <16 x float> %v16f1, %v16f2
; CHECK: %4 = fadd fast double %d1, %d2
; CHECK: %5 = fadd fast <2 x double> %v2d1, %v2d2
; CHECK: %6 = fadd fast <4 x double> %v4d1, %v4d2
; CHECK: %7 = fadd fast <8 x double> %v8d1, %v8d2
define void @svml_fadd(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                       float %f2, <4 x float> %v4f2, <8 x float> %v8f2, <16 x float> %v16f2,
                       double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                       double %d2, <2 x double> %v2d2, <4 x double> %v4d2, <8 x double> %v8d2) {
entry:
  %t0_0 = call fast float @llvm.fpbuiltin.fadd.f32(float %f1, float %f2) #0
  %t1_0 = call fast <4 x float> @llvm.fpbuiltin.fadd.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #0
  %t2_0 = call fast <8 x float> @llvm.fpbuiltin.fadd.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #0
  %t3_0 = call fast <16 x float> @llvm.fpbuiltin.fadd.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #0
  %t4_0 = call fast double @llvm.fpbuiltin.fadd.f64(double %d1, double %d2) #0
  %t5_0 = call fast <2 x double> @llvm.fpbuiltin.fadd.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  %t6_0 = call fast <4 x double> @llvm.fpbuiltin.fadd.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #0
  %t7_0 = call fast <8 x double> @llvm.fpbuiltin.fadd.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #0
  ret void
}

declare float @llvm.fpbuiltin.fadd.f32(float, float)
declare <4 x float> @llvm.fpbuiltin.fadd.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.fpbuiltin.fadd.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.fpbuiltin.fadd.v16f32(<16 x float>, <16 x float>)
declare double @llvm.fpbuiltin.fadd.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.fadd.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.fpbuiltin.fadd.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.fpbuiltin.fadd.v8f64(<8 x double>, <8 x double>)

; CHECK-LABEL: @svml_fsub
; CHECK: %0 = fsub fast float %f1, %f2
; CHECK: %1 = fsub fast <4 x float> %v4f1, %v4f2
; CHECK: %2 = fsub fast <8 x float> %v8f1, %v8f2
; CHECK: %3 = fsub fast <16 x float> %v16f1, %v16f2
; CHECK: %4 = fsub fast double %d1, %d2
; CHECK: %5 = fsub fast <2 x double> %v2d1, %v2d2
; CHECK: %6 = fsub fast <4 x double> %v4d1, %v4d2
; CHECK: %7 = fsub fast <8 x double> %v8d1, %v8d2
define void @svml_fsub(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                       float %f2, <4 x float> %v4f2, <8 x float> %v8f2, <16 x float> %v16f2,
                       double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                       double %d2, <2 x double> %v2d2, <4 x double> %v4d2, <8 x double> %v8d2) {
entry:
  %t0_0 = call fast float @llvm.fpbuiltin.fsub.f32(float %f1, float %f2) #0
  %t1_0 = call fast <4 x float> @llvm.fpbuiltin.fsub.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #0
  %t2_0 = call fast <8 x float> @llvm.fpbuiltin.fsub.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #0
  %t3_0 = call fast <16 x float> @llvm.fpbuiltin.fsub.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #0
  %t4_0 = call fast double @llvm.fpbuiltin.fsub.f64(double %d1, double %d2) #0
  %t5_0 = call fast <2 x double> @llvm.fpbuiltin.fsub.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  %t6_0 = call fast <4 x double> @llvm.fpbuiltin.fsub.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #0
  %t7_0 = call fast <8 x double> @llvm.fpbuiltin.fsub.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #0
  ret void
}

declare float @llvm.fpbuiltin.fsub.f32(float, float)
declare <4 x float> @llvm.fpbuiltin.fsub.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.fpbuiltin.fsub.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.fpbuiltin.fsub.v16f32(<16 x float>, <16 x float>)
declare double @llvm.fpbuiltin.fsub.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.fsub.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.fpbuiltin.fsub.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.fpbuiltin.fsub.v8f64(<8 x double>, <8 x double>)

; CHECK-LABEL: @svml_fmul
; CHECK: %0 = fmul fast float %f1, %f2
; CHECK: %1 = fmul fast <4 x float> %v4f1, %v4f2
; CHECK: %2 = fmul fast <8 x float> %v8f1, %v8f2
; CHECK: %3 = fmul fast <16 x float> %v16f1, %v16f2
; CHECK: %4 = fmul fast double %d1, %d2
; CHECK: %5 = fmul fast <2 x double> %v2d1, %v2d2
; CHECK: %6 = fmul fast <4 x double> %v4d1, %v4d2
; CHECK: %7 = fmul fast <8 x double> %v8d1, %v8d2
define void @svml_fmul(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                       float %f2, <4 x float> %v4f2, <8 x float> %v8f2, <16 x float> %v16f2,
                       double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                       double %d2, <2 x double> %v2d2, <4 x double> %v4d2, <8 x double> %v8d2) {
entry:
  %t0_0 = call fast float @llvm.fpbuiltin.fmul.f32(float %f1, float %f2) #0
  %t1_0 = call fast <4 x float> @llvm.fpbuiltin.fmul.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #0
  %t2_0 = call fast <8 x float> @llvm.fpbuiltin.fmul.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #0
  %t3_0 = call fast <16 x float> @llvm.fpbuiltin.fmul.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #0
  %t4_0 = call fast double @llvm.fpbuiltin.fmul.f64(double %d1, double %d2) #0
  %t5_0 = call fast <2 x double> @llvm.fpbuiltin.fmul.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  %t6_0 = call fast <4 x double> @llvm.fpbuiltin.fmul.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #0
  %t7_0 = call fast <8 x double> @llvm.fpbuiltin.fmul.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #0
  ret void
}

declare float @llvm.fpbuiltin.fmul.f32(float, float)
declare <4 x float> @llvm.fpbuiltin.fmul.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.fpbuiltin.fmul.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.fpbuiltin.fmul.v16f32(<16 x float>, <16 x float>)
declare double @llvm.fpbuiltin.fmul.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.fmul.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.fpbuiltin.fmul.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.fpbuiltin.fmul.v8f64(<8 x double>, <8 x double>)

; CHECK-LABEL: @svml_fdiv
; CHECK: %0 = fdiv fast double %d1, %d2
; CHECK: %1 = fdiv fast <2 x double> %v2d1, %v2d2
; CHECK: %2 = fdiv fast <4 x double> %v4d1, %v4d2
; CHECK: %3 = fdiv fast <8 x double> %v8d1, %v8d2
define void @svml_fdiv(double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                       double %d2, <2 x double> %v2d2, <4 x double> %v4d2, <8 x double> %v8d2) {
entry:
  %t0_0 = call fast double @llvm.fpbuiltin.fdiv.f64(double %d1, double %d2) #0
  %t1_0 = call fast <2 x double> @llvm.fpbuiltin.fdiv.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  %t2_0 = call fast <4 x double> @llvm.fpbuiltin.fdiv.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #0
  %t3_0 = call fast <8 x double> @llvm.fpbuiltin.fdiv.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #0
  ret void
}

declare double @llvm.fpbuiltin.fdiv.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.fdiv.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.fpbuiltin.fdiv.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.fpbuiltin.fdiv.v8f64(<8 x double>, <8 x double>)

; CHECK-LABEL: @svml_frem
; CHECK: %0 = frem fast float %f1, %f2
; CHECK: %1 = frem fast <4 x float> %v4f1, %v4f2
; CHECK: %2 = frem fast <8 x float> %v8f1, %v8f2
; CHECK: %3 = frem fast <16 x float> %v16f1, %v16f2
; CHECK: %4 = frem fast double %d1, %d2
; CHECK: %5 = frem fast <2 x double> %v2d1, %v2d2
; CHECK: %6 = frem fast <4 x double> %v4d1, %v4d2
; CHECK: %7 = frem fast <8 x double> %v8d1, %v8d2
define void @svml_frem(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                       float %f2, <4 x float> %v4f2, <8 x float> %v8f2, <16 x float> %v16f2,
                       double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                       double %d2, <2 x double> %v2d2, <4 x double> %v4d2, <8 x double> %v8d2) {
entry:
  %t0_0 = call fast float @llvm.fpbuiltin.frem.f32(float %f1, float %f2) #0
  %t1_0 = call fast <4 x float> @llvm.fpbuiltin.frem.v4f32(<4 x float> %v4f1, <4 x float> %v4f2) #0
  %t2_0 = call fast <8 x float> @llvm.fpbuiltin.frem.v8f32(<8 x float> %v8f1, <8 x float> %v8f2) #0
  %t3_0 = call fast <16 x float> @llvm.fpbuiltin.frem.v16f32(<16 x float> %v16f1, <16 x float> %v16f2) #0
  %t4_0 = call fast double @llvm.fpbuiltin.frem.f64(double %d1, double %d2) #0
  %t5_0 = call fast <2 x double> @llvm.fpbuiltin.frem.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  %t6_0 = call fast <4 x double> @llvm.fpbuiltin.frem.v4f64(<4 x double> %v4d1, <4 x double> %v4d2) #0
  %t7_0 = call fast <8 x double> @llvm.fpbuiltin.frem.v8f64(<8 x double> %v8d1, <8 x double> %v8d2) #0
  ret void
}

declare float @llvm.fpbuiltin.frem.f32(float, float)
declare <4 x float> @llvm.fpbuiltin.frem.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.fpbuiltin.frem.v8f32(<8 x float>, <8 x float>)
declare <16 x float> @llvm.fpbuiltin.frem.v16f32(<16 x float>, <16 x float>)
declare double @llvm.fpbuiltin.frem.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.frem.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.fpbuiltin.frem.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.fpbuiltin.frem.v8f64(<8 x double>, <8 x double>)

; CHECK-LABEL: @svml_sqrt
; CHECK: %0 = call double @llvm.sqrt.f64(double %d)
; CHECK: %1 = call <2 x double> @llvm.sqrt.v2f64(<2 x double> %v2d)
; CHECK: %2 = call <4 x double> @llvm.sqrt.v4f64(<4 x double> %v4d)
; CHECK: %3 = call <8 x double> @llvm.sqrt.v8f64(<8 x double> %v8d)
define void @svml_sqrt(double %d, <2 x double> %v2d, <4 x double> %v4d, <8 x double> %v8d) {
entry:
  %t4_0 = call double @llvm.fpbuiltin.sqrt.f64(double %d) #0
  %t5_0 = call <2 x double> @llvm.fpbuiltin.sqrt.v2f64(<2 x double> %v2d) #0
  %t6_0 = call <4 x double> @llvm.fpbuiltin.sqrt.v4f64(<4 x double> %v4d) #0
  %t7_0 = call <8 x double> @llvm.fpbuiltin.sqrt.v8f64(<8 x double> %v8d) #0
  ret void
}

declare double @llvm.fpbuiltin.sqrt.f64(double)
declare <2 x double> @llvm.fpbuiltin.sqrt.v2f64(<2 x double>)
declare <4 x double> @llvm.fpbuiltin.sqrt.v4f64(<4 x double>)
declare <8 x double> @llvm.fpbuiltin.sqrt.v8f64(<8 x double>)

; CHECK-LABEL: @svml_ldexp
; CHECK: %0 = call fast float @llvm.ldexp.f32.i32(float %f1, i32 %f2)
; CHECK: %1 = call fast <4 x float> @llvm.ldexp.v4f32.v4i32(<4 x float> %v4f1, <4 x i32> %v4f2)
; CHECK: %2 = call fast <8 x float> @llvm.ldexp.v8f32.v8i32(<8 x float> %v8f1, <8 x i32> %v8f2)
; CHECK: %3 = call fast <16 x float> @llvm.ldexp.v16f32.v16i32(<16 x float> %v16f1, <16 x i32> %v16f2)
; CHECK: %4 = call fast double @llvm.ldexp.f64.i32(double %d1, i32 %d2)
; CHECK: %5 = call fast <2 x double> @llvm.ldexp.v2f64.v2i32(<2 x double> %v2d1, <2 x i32> %v2d2)
; CHECK: %6 = call fast <4 x double> @llvm.ldexp.v4f64.v4i32(<4 x double> %v4d1, <4 x i32> %v4d2)
; CHECK: %7 = call fast <8 x double> @llvm.ldexp.v8f64.v8i32(<8 x double> %v8d1, <8 x i32> %v8d2)
define void @svml_ldexp(float %f1, <4 x float> %v4f1, <8 x float> %v8f1, <16 x float> %v16f1,
                        i32 %f2, <4 x i32> %v4f2, <8 x i32> %v8f2, <16 x i32> %v16f2,
                        double %d1, <2 x double> %v2d1, <4 x double> %v4d1, <8 x double> %v8d1,
                        i32 %d2, <2 x i32> %v2d2, <4 x i32> %v4d2, <8 x i32> %v8d2) {
entry:
  %t0_0 = call fast float @llvm.fpbuiltin.ldexp.f32.i32(float %f1, i32 %f2) #0
  %t1_0 = call fast <4 x float> @llvm.fpbuiltin.ldexp.v4f32.v4i32(<4 x float> %v4f1, <4 x i32> %v4f2) #0
  %t2_0 = call fast <8 x float> @llvm.fpbuiltin.ldexp.v8f32.v8i32(<8 x float> %v8f1, <8 x i32> %v8f2) #0
  %t3_0 = call fast <16 x float> @llvm.fpbuiltin.ldexp.v16f32.v16i32(<16 x float> %v16f1, <16 x i32> %v16f2) #0
  %t4_0 = call fast double @llvm.fpbuiltin.ldexp.f64.i32(double %d1, i32 %d2) #0
  %t5_0 = call fast <2 x double> @llvm.fpbuiltin.ldexp.v2f64.v2i32(<2 x double> %v2d1, <2 x i32> %v2d2) #0
  %t6_0 = call fast <4 x double> @llvm.fpbuiltin.ldexp.v4f64.v4i32(<4 x double> %v4d1, <4 x i32> %v4d2) #0
  %t7_0 = call fast <8 x double> @llvm.fpbuiltin.ldexp.v8f64.v8i32(<8 x double> %v8d1, <8 x i32> %v8d2) #0
  ret void
}

declare float @llvm.fpbuiltin.ldexp.f32.i32(float, i32)
declare <4 x float> @llvm.fpbuiltin.ldexp.v4f32.v4i32(<4 x float>, <4 x i32>)
declare <8 x float> @llvm.fpbuiltin.ldexp.v8f32.v8i32(<8 x float>, <8 x i32>)
declare <16 x float> @llvm.fpbuiltin.ldexp.v16f32.v16i32(<16 x float>, <16 x i32>)
declare double @llvm.fpbuiltin.ldexp.f64.i32(double, i32)
declare <2 x double> @llvm.fpbuiltin.ldexp.v2f64.v2i32(<2 x double>, <2 x i32>)
declare <4 x double> @llvm.fpbuiltin.ldexp.v4f64.v4i32(<4 x double>, <4 x i32>)
declare <8 x double> @llvm.fpbuiltin.ldexp.v8f64.v8i32(<8 x double>, <8 x i32>)

attributes #0 = { "fpbuiltin-max-error"="0.5" }
