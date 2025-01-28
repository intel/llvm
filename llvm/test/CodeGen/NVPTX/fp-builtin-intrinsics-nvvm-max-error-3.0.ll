; RUN: opt -fpbuiltin-fn-selection -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: @test_fdiv
; CHECK: %{{.*}} = call float @llvm.nvvm.div.approx.f(float %{{.*}}, float %{{.*}})
; CHECK: %{{.*}} = fdiv <2 x float> %{{.*}}, %{{.*}}
define void @test_fdiv(float %d1, <2 x float> %v2d1,
                       float %d2, <2 x float> %v2d2) {
entry:
  %t0 = call float @llvm.fpbuiltin.fdiv.f32(float %d1, float %d2) #0
  %t1 = call <2 x float> @llvm.fpbuiltin.fdiv.v2f32(<2 x float> %v2d1, <2 x float> %v2d2) #0
  ret void
}

; CHECK-LABEL: @test_fdiv_fast
; CHECK: %{{.*}} = call fast float @llvm.nvvm.div.approx.ftz.f(float %{{.*}}, float %{{.*}})
; CHECK: %{{.*}} = fdiv fast <2 x float> %{{.*}}, %{{.*}}
define void @test_fdiv_fast(float %d1, <2 x float> %v2d1,
                            float %d2, <2 x float> %v2d2) {
entry:
  %t0 = call fast float @llvm.fpbuiltin.fdiv.f32(float %d1, float %d2) #0
  %t1 = call fast <2 x float> @llvm.fpbuiltin.fdiv.v2f32(<2 x float> %v2d1, <2 x float> %v2d2) #0
  ret void
}

declare float @llvm.fpbuiltin.fdiv.f32(float, float)
declare <2 x float> @llvm.fpbuiltin.fdiv.v2f32(<2 x float>, <2 x float>)

; CHECK-LABEL: @test_fdiv_double
; CHECK: %{{.*}} = fdiv double %{{.*}}, %{{.*}}
; CHECK: %{{.*}} = fdiv <2 x double> %{{.*}}, %{{.*}}
define void @test_fdiv_double(double %d1, <2 x double> %v2d1,
                              double %d2, <2 x double> %v2d2) {
entry:
  %t0 = call double @llvm.fpbuiltin.fdiv.f64(double %d1, double %d2) #0
  %t1 = call <2 x double> @llvm.fpbuiltin.fdiv.v2f64(<2 x double> %v2d1, <2 x double> %v2d2) #0
  ret void
}

declare double @llvm.fpbuiltin.fdiv.f64(double, double)
declare <2 x double> @llvm.fpbuiltin.fdiv.v2f64(<2 x double>, <2 x double>)

; CHECK-LABEL: @test_sqrt
; CHECK: %{{.*}} = call float @llvm.nvvm.sqrt.approx.f(float %{{.*}})
; CHECK: %{{.*}} = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %{{.*}})
define void @test_sqrt(float %d, <2 x float> %v2d, <4 x float> %v4d) {
entry:
  %t0 = call float @llvm.fpbuiltin.sqrt.f32(float %d) #0
  %t1 = call <2 x float> @llvm.fpbuiltin.sqrt.v2f32(<2 x float> %v2d) #0
  ret void
}

; CHECK-LABEL: @test_sqrt_fast
; CHECK: %{{.*}} = call fast float @llvm.nvvm.sqrt.approx.ftz.f(float %{{.*}})
; CHECK: %{{.*}} = call fast <2 x float> @llvm.sqrt.v2f32(<2 x float> %{{.*}})
define void @test_sqrt_fast(float %d, <2 x float> %v2d, <4 x float> %v4d) {
entry:
  %t0 = call fast float @llvm.fpbuiltin.sqrt.f32(float %d) #0
  %t1 = call fast <2 x float> @llvm.fpbuiltin.sqrt.v2f32(<2 x float> %v2d) #0
  ret void
}

declare float @llvm.fpbuiltin.sqrt.f32(float)
declare <2 x float> @llvm.fpbuiltin.sqrt.v2f32(<2 x float>)

; CHECK-LABEL: @test_sqrt_double
; CHECK: %{{.*}} = call double @llvm.sqrt.f64(double %{{.*}})
; CHECK: %{{.*}} = call <2 x double> @llvm.sqrt.v2f64(<2 x double> %{{.*}})
define void @test_sqrt_double(double %d, <2 x double> %v2d) {
entry:
  %t0 = call double @llvm.fpbuiltin.sqrt.f64(double %d) #0
  %t1 = call <2 x double> @llvm.fpbuiltin.sqrt.v2f64(<2 x double> %v2d) #0
  ret void
}

declare double @llvm.fpbuiltin.sqrt.f64(double)
declare <2 x double> @llvm.fpbuiltin.sqrt.v2f64(<2 x double>)

attributes #0 = { "fpbuiltin-max-error"="3.0" }
