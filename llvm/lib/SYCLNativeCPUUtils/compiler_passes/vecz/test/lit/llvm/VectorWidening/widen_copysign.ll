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

; REQUIRES: llvm-12+
; RUN: veczc -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

declare float @llvm.copysign.f32(float, float)
declare <2 x float> @llvm.copysign.v2f32(<2 x float>, <2 x float>)

define spir_kernel void @copysignff(float* %pa, float* %pb, float* %pc) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %a = getelementptr float, float* %pa, i64 %idx
  %b = getelementptr float, float* %pb, i64 %idx
  %c = getelementptr float, float* %pc, i64 %idx
  %la = load float, float* %a, align 16
  %lb = load float, float* %b, align 16
  %res = call float @llvm.copysign.f32(float %la, float %lb)
  store float %res, float* %c, align 16
  ret void
}

define spir_kernel void @copysignvf(<2 x float>* %pa, <2 x float>* %pb, <2 x float>* %pc) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %a = getelementptr <2 x float>, <2 x float>* %pa, i64 %idx
  %b = getelementptr <2 x float>, <2 x float>* %pb, i64 %idx
  %c = getelementptr <2 x float>, <2 x float>* %pc, i64 %idx
  %la = load <2 x float>, <2 x float>* %a, align 16
  %lb = load <2 x float>, <2 x float>* %b, align 16
  %res = call <2 x float> @llvm.copysign.v2f32(<2 x float> %la, <2 x float> %lb)
  store <2 x float> %res, <2 x float>* %c, align 16
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_copysignff(ptr %pa, ptr %pb, ptr %pc)
; CHECK: entry:
; CHECK: %idx = call i64 @__mux_get_global_id(i32 0)
; CHECK: %a = getelementptr float, ptr %pa, i64 %idx
; CHECK: %b = getelementptr float, ptr %pb, i64 %idx
; CHECK: %c = getelementptr float, ptr %pc, i64 %idx
; CHECK: [[T0:%.*]] = load <4 x float>, ptr %a, align 4
; CHECK: [[T1:%.*]] = load <4 x float>, ptr %b, align 4
; CHECK: %res1 = call <4 x float> @llvm.copysign.v4f32(<4 x float> [[T0]], <4 x float> [[T1]])
; CHECK: store <4 x float> %res1, ptr %c, align 4
; CHECK: ret void

; CHECK: define spir_kernel void @__vecz_v4_copysignvf(ptr %pa, ptr %pb, ptr %pc)
; CHECK: entry:
; CHECK: %idx = call i64 @__mux_get_global_id(i32 0)
; CHECK: %a = getelementptr <2 x float>, ptr %pa, i64 %idx
; CHECK: %b = getelementptr <2 x float>, ptr %pb, i64 %idx
; CHECK: %c = getelementptr <2 x float>, ptr %pc, i64 %idx
; CHECK: [[T0:%.*]] = load <8 x float>, ptr %a, align 4
; CHECK: [[T1:%.*]] = load <8 x float>, ptr %b, align 4
; CHECK: %res1 = call <8 x float> @llvm.copysign.v8f32(<8 x float> [[T0]], <8 x float> [[T1]])
; CHECK: store <8 x float> %res1, ptr %c, align 8
; CHECK: ret void
