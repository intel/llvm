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

; RUN: veczc -k fmin_vector_scalar -vecz-simd-width=4 -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

; Function Attrs: nounwind readnone
declare spir_func <4 x float> @_Z4fminDv4_ff(<4 x float>, float)

; Note that we have to declare the scalar version, because when we vectorize
; an already-vector builtin, we have to scalarize it first. This is the case
; even for Vector Widening, where we don't actually create a call to the
; scalar version, but we retrieve the wide version via the scalar version,
; so the declaration still needs to exist.

; Function Attrs: inlinehint nounwind readnone
declare spir_func float @_Z4fminff(float, float)

; Function Attrs: inlinehint nounwind readnone
declare spir_func <16 x float> @_Z4fminDv16_fS_(<16 x float>, <16 x float>)

define spir_kernel void @fmin_vector_scalar(<4 x float>* %pa, float* %pb, <4 x float>* %pd) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %a = getelementptr <4 x float>, <4 x float>* %pa, i64 %idx
  %b = getelementptr float, float* %pb, i64 %idx
  %d = getelementptr <4 x float>, <4 x float>* %pd, i64 %idx
  %la = load <4 x float>, <4 x float>* %a, align 16
  %lb = load float, float* %b, align 4
  %res = tail call spir_func <4 x float> @_Z4fminDv4_ff(<4 x float> %la, float %lb)
  store <4 x float> %res, <4 x float>* %d, align 16
  ret void
}


; CHECK: define spir_kernel void @__vecz_v4_fmin_vector_scalar(ptr %pa, ptr %pb, ptr %pd)
; CHECK: entry:

; It checks that the fmin builtin gets widened by a factor of 4, while its
; scalar operand is sub-splatted to the required <16 x float>.
; CHECK: %[[LDA:.+]] = load <16 x float>, ptr %{{.+}}
; CHECK: %[[LDB:.+]] = load <4 x float>, ptr %{{.+}}
; CHECK: %[[SPL:.+]] = shufflevector <4 x float> %[[LDB]], <4 x float> {{undef|poison}}, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 3>
; CHECK: %[[RES:.+]] = call <16 x float> @llvm.minnum.v16f32(<16 x float> %[[LDA]], <16 x float> %[[SPL]])
; CHECK: store <16 x float> %[[RES]], ptr %{{.+}}

; CHECK: ret void
