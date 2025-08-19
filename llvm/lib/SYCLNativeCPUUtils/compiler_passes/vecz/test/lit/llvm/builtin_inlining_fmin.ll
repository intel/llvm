; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: veczc -vecz-passes=builtin-inlining -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @fminff(float %a, float %b, float* %c) {
entry:
  %min = call spir_func float @_Z4fminff(float %a, float %b)
  store float %min, float* %c, align 4
  ret void
}

define spir_kernel void @fminvf(<2 x float> %a, float %b, <2 x float>* %c) {
entry:
  %min = call spir_func <2 x float> @_Z4fminDv2_ff(<2 x float> %a, float %b)
  store <2 x float> %min, <2 x float>* %c, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)

declare spir_func float @_Z4fminff(float, float)
declare spir_func <2 x float> @_Z4fminDv2_ff(<2 x float>, float)

; CHECK: define spir_kernel void @__vecz_v4_fminff(float %a, float %b, ptr %c)
; CHECK: entry:
; CHECK: %0 = call float @llvm.minnum.f32(float %a, float %b)
; CHECK: store float %0, ptr %c, align 4
; CHECK: ret void

; CHECK: define spir_kernel void @__vecz_v4_fminvf(<2 x float> %a, float %b, ptr %c)
; CHECK: entry:
; CHECK: %.splatinsert = insertelement <2 x float> {{.*}}, float %b, {{(i32|i64)}} 0
; CHECK: %.splat = shufflevector <2 x float> %.splatinsert, <2 x float> {{.*}}, <2 x i32> zeroinitializer
; CHECK: %0 = call <2 x float> @llvm.minnum.v2f32(<2 x float> %a, <2 x float> %.splat)
; CHECK: store <2 x float> %0, ptr %c, align 4
; CHECK: ret void
