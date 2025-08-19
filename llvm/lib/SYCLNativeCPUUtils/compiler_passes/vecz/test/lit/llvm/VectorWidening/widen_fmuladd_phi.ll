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

; RUN: veczc -k test_calls -vecz-passes=packetizer -vecz-simd-width=8 -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @test_calls(<4 x float>* %pa, <4 x float>* %pb, <4 x float>* %pc, <4 x float>* %pd) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %a = getelementptr <4 x float>, <4 x float>* %pa, i64 %idx
  %b = getelementptr <4 x float>, <4 x float>* %pb, i64 %idx
  %c = getelementptr <4 x float>, <4 x float>* %pc, i64 %idx
  %d = getelementptr <4 x float>, <4 x float>* %pd, i64 %idx
  %la = load <4 x float>, <4 x float>* %a, align 16
  %lb = load <4 x float>, <4 x float>* %b, align 16
  %lc = load <4 x float>, <4 x float>* %c, align 16
  br label %loop

loop:
  %n = phi i32 [ %dec, %loop ], [ 10, %entry ]
  %acc = phi <4 x float> [ %fma, %loop ], [ %la, %entry ]
  %fma = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %acc, <4 x float> %lb, <4 x float> %lc)
  %dec = sub i32 %n, 1
  %cmp = icmp ne i32 %dec, 0
  br i1 %cmp, label %loop, label %end

end:
  store <4 x float> %fma, <4 x float>* %d, align 16
  ret void
}

declare <4x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>)

; CHECK: define spir_kernel void @__vecz_v8_test_calls(ptr %pa, ptr %pb, ptr %pc, ptr %pd)
; CHECK: entry:

; It checks that the fmuladd intrinsic of <4 x float> gets widened by a factor of 8,
; to produce a PAIR of <16 x float>s.
; CHECK: %[[LDA0:.+]] = load <16 x float>, ptr %{{.+}}, align 16
; CHECK: %[[LDA1:.+]] = load <16 x float>, ptr %{{.+}}, align 16
; CHECK: %[[LDB0:.+]] = load <16 x float>, ptr %{{.+}}, align 16
; CHECK: %[[LDB1:.+]] = load <16 x float>, ptr %{{.+}}, align 16
; CHECK: %[[LDC0:.+]] = load <16 x float>, ptr %{{.+}}, align 16
; CHECK: %[[LDC1:.+]] = load <16 x float>, ptr %{{.+}}, align 16

; CHECK: loop:
; CHECK: %[[ACC0:.+]] = phi <16 x float> [ %[[FMA0:.+]], %loop ], [ %[[LDA0]], %entry ]
; CHECK: %[[ACC1:.+]] = phi <16 x float> [ %[[FMA1:.+]], %loop ], [ %[[LDA1]], %entry ]

; CHECK: %[[FMA0]] = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %[[ACC0]], <16 x float> %[[LDB0]], <16 x float> %[[LDC0]])
; CHECK: %[[FMA1]] = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %[[ACC1]], <16 x float> %[[LDB1]], <16 x float> %[[LDC1]])

; CHECK: end:
; CHECK: store <16 x float> %[[FMA0]], ptr %{{.+}}, align 16
; CHECK: store <16 x float> %[[FMA1]], ptr %{{.+}}, align 16

; CHECK: ret void
