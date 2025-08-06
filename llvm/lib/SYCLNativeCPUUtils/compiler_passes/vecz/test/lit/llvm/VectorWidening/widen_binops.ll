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

; RUN: veczc -k widen_binops -vecz-passes=packetizer -vecz-simd-width=8 -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @widen_binops(<4 x i32>* %pa, <4 x i32>* %pb, <4 x i64>* %pd) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %a = getelementptr <4 x i32>, <4 x i32>* %pa, i64 %idx
  %b = getelementptr <4 x i32>, <4 x i32>* %pb, i64 %idx
  %d = getelementptr <4 x i64>, <4 x i64>* %pd, i64 %idx
  %la = load <4 x i32>, <4 x i32>* %a, align 16
  %lb = load <4 x i32>, <4 x i32>* %b, align 16
  %xa = zext <4 x i32> %la to <4 x i64>
  %xb = zext <4 x i32> %lb to <4 x i64>
  %add = add nuw nsw <4 x i64> %xa, %xb
  store <4 x i64> %add, <4 x i64>* %d, align 16
  ret void
}

; CHECK: define spir_kernel void @__vecz_v8_widen_binops(ptr %pa, ptr %pb, ptr %pd)
; CHECK: entry:

; It checks that the zexts and add of <4 x i32> gets widened by a factor of 8,
; to produce PAIRs of <16 x i32>s.
; CHECK: %[[LDA0:.+]] = load <16 x i32>, ptr %{{.+}}, align 16
; CHECK: %[[LDA1:.+]] = load <16 x i32>, ptr %{{.+}}, align 16
; CHECK: %[[LDB0:.+]] = load <16 x i32>, ptr %{{.+}}, align 16
; CHECK: %[[LDB1:.+]] = load <16 x i32>, ptr %{{.+}}, align 16
; CHECK: %[[XA0:.+]] = zext <16 x i32> %[[LDA0]] to <16 x i64>
; CHECK: %[[XA1:.+]] = zext <16 x i32> %[[LDA1]] to <16 x i64>
; CHECK: %[[XB0:.+]] = zext <16 x i32> %[[LDB0]] to <16 x i64>
; CHECK: %[[XB1:.+]] = zext <16 x i32> %[[LDB1]] to <16 x i64>
; CHECK: %[[ADD0:.+]] = add nuw nsw <16 x i64> %[[XA0]], %[[XB0]]
; CHECK: %[[ADD1:.+]] = add nuw nsw <16 x i64> %[[XA1]], %[[XB1]]
; CHECK: store <16 x i64> %[[ADD0]], ptr %{{.+}}
; CHECK: store <16 x i64> %[[ADD1]], ptr %{{.+}}

; CHECK: ret void
