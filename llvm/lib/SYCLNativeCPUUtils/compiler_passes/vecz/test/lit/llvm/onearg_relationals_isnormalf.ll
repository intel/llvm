; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: %veczc -k test_isnormalf -vecz-simd-width=4 -S < %s | %filecheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func i32 @_Z5isinfd(double)
declare spir_func i32 @_Z5isinff(float)
declare spir_func i32 @_Z5isnand(double)
declare spir_func i32 @_Z5isnanf(float)
declare spir_func i32 @_Z7signbitd(double)
declare spir_func i32 @_Z7signbitf(float)
declare spir_func i32 @_Z8isfinited(double)
declare spir_func i32 @_Z8isfinitef(float)
declare spir_func i32 @_Z8isnormald(double)
declare spir_func i32 @_Z8isnormalf(float)
declare spir_func <4 x i32> @_Z5isinfDv4_f(<4 x float>)
declare spir_func <4 x i32> @_Z5isnanDv4_f(<4 x float>)
declare spir_func <4 x i32> @_Z7signbitDv4_f(<4 x float>)
declare spir_func <4 x i32> @_Z8isfiniteDv4_f(<4 x float>)
declare spir_func <4 x i32> @_Z8isnormalDv4_f(<4 x float>)
declare spir_func <4 x i64> @_Z5isinfDv4_d(<4 x double>)
declare spir_func <4 x i64> @_Z5isnanDv4_d(<4 x double>)
declare spir_func <4 x i64> @_Z7signbitDv4_d(<4 x double>)
declare spir_func <4 x i64> @_Z8isfiniteDv4_d(<4 x double>)
declare spir_func <4 x i64> @_Z8isnormalDv4_d(<4 x double>)

define spir_kernel void @test_isfinitef(float addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 @_Z8isfinitef(float %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_isfinited(double addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds double, double addrspace(1)* %in, i64 %call
  %0 = load double, double addrspace(1)* %arrayidx, align 8
  %call1 = call spir_func i32 @_Z8isfinited(double %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_isfiniteDv4_f(<4 x float> addrspace(1)* %in, <4 x i32> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 16
  %call1 = call spir_func <4 x i32> @_Z8isfiniteDv4_f(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out, i64 %call
  store <4 x i32> %call1, <4 x i32> addrspace(1)* %arrayidx2, align 16
  ret void
}

define spir_kernel void @test_isfiniteDv4_d(<4 x double> addrspace(1)* %in, <4 x i64> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %in, i64 %call
  %0 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx, align 32
  %call1 = call spir_func <4 x i64> @_Z8isfiniteDv4_d(<4 x double> %0)
  %arrayidx2 = getelementptr inbounds <4 x i64>, <4 x i64> addrspace(1)* %out, i64 %call
  store <4 x i64> %call1, <4 x i64> addrspace(1)* %arrayidx2, align 32
  ret void
}

define spir_kernel void @test_isinff(float addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 @_Z5isinff(float %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_isinfd(double addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds double, double addrspace(1)* %in, i64 %call
  %0 = load double, double addrspace(1)* %arrayidx, align 8
  %call1 = call spir_func i32 @_Z5isinfd(double %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_isinfDv4_f(<4 x float> addrspace(1)* %in, <4 x i32> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 16
  %call1 = call spir_func <4 x i32> @_Z5isinfDv4_f(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out, i64 %call
  store <4 x i32> %call1, <4 x i32> addrspace(1)* %arrayidx2, align 16
  ret void
}

define spir_kernel void @test_isinfDv4_d(<4 x double> addrspace(1)* %in, <4 x i64> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %in, i64 %call
  %0 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx, align 32
  %call1 = call spir_func <4 x i64> @_Z5isinfDv4_d(<4 x double> %0)
  %arrayidx2 = getelementptr inbounds <4 x i64>, <4 x i64> addrspace(1)* %out, i64 %call
  store <4 x i64> %call1, <4 x i64> addrspace(1)* %arrayidx2, align 32
  ret void
}

define spir_kernel void @test_isnormalf(float addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 @_Z8isnormalf(float %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_isnormald(double addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds double, double addrspace(1)* %in, i64 %call
  %0 = load double, double addrspace(1)* %arrayidx, align 8
  %call1 = call spir_func i32 @_Z8isnormald(double %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_isnormalDv4_f(<4 x float> addrspace(1)* %in, <4 x i32> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 16
  %call1 = call spir_func <4 x i32> @_Z8isnormalDv4_f(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out, i64 %call
  store <4 x i32> %call1, <4 x i32> addrspace(1)* %arrayidx2, align 16
  ret void
}

define spir_kernel void @test_isnormalDv4_d(<4 x double> addrspace(1)* %in, <4 x i64> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %in, i64 %call
  %0 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx, align 32
  %call1 = call spir_func <4 x i64> @_Z8isnormalDv4_d(<4 x double> %0)
  %arrayidx2 = getelementptr inbounds <4 x i64>, <4 x i64> addrspace(1)* %out, i64 %call
  store <4 x i64> %call1, <4 x i64> addrspace(1)* %arrayidx2, align 32
  ret void
}

define spir_kernel void @test_isnanf(float addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 @_Z5isnanf(float %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_isnand(double addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds double, double addrspace(1)* %in, i64 %call
  %0 = load double, double addrspace(1)* %arrayidx, align 8
  %call1 = call spir_func i32 @_Z5isnand(double %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_isnanDv4_f(<4 x float> addrspace(1)* %in, <4 x i32> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 16
  %call1 = call spir_func <4 x i32> @_Z5isnanDv4_f(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out, i64 %call
  store <4 x i32> %call1, <4 x i32> addrspace(1)* %arrayidx2, align 16
  ret void
}

define spir_kernel void @test_isnanDv4_d(<4 x double> addrspace(1)* %in, <4 x i64> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %in, i64 %call
  %0 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx, align 32
  %call1 = call spir_func <4 x i64> @_Z5isnanDv4_d(<4 x double> %0)
  %arrayidx2 = getelementptr inbounds <4 x i64>, <4 x i64> addrspace(1)* %out, i64 %call
  store <4 x i64> %call1, <4 x i64> addrspace(1)* %arrayidx2, align 32
  ret void
}

define spir_kernel void @test_signbitf(float addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 @_Z7signbitf(float %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_signbitd(double addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds double, double addrspace(1)* %in, i64 %call
  %0 = load double, double addrspace(1)* %arrayidx, align 8
  %call1 = call spir_func i32 @_Z7signbitd(double %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
}

define spir_kernel void @test_signbitDv4_f(<4 x float> addrspace(1)* %in, <4 x i32> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 16
  %call1 = call spir_func <4 x i32> @_Z7signbitDv4_f(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out, i64 %call
  store <4 x i32> %call1, <4 x i32> addrspace(1)* %arrayidx2, align 16
  ret void
}

define spir_kernel void @test_signbitDv4_d(<4 x double> addrspace(1)* %in, <4 x i64> addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %in, i64 %call
  %0 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx, align 32
  %call1 = call spir_func <4 x i64> @_Z7signbitDv4_d(<4 x double> %0)
  %arrayidx2 = getelementptr inbounds <4 x i64>, <4 x i64> addrspace(1)* %out, i64 %call
  store <4 x i64> %call1, <4 x i64> addrspace(1)* %arrayidx2, align 32
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_test_isnormalf
; CHECK: and <4 x i32>
; CHECK: add nsw <4 x i32>
; CHECK: icmp ult <4 x i32>
; CHECK: zext <4 x i1>
; CHECK: ret void
