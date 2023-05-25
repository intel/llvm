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

; RUN: veczc -k test_calls -vecz-passes=packetizer -vecz-simd-width=8 -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @test_calls(<4 x float>* %pa, <4 x float>* %pb, <4 x float>* %pc, <4 x float>* %pd) {
entry:
  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
  %idx2 = shl i64 %idx, 1
  %a = getelementptr <4 x float>, <4 x float>* %pa, i64 %idx2
  %b = getelementptr <4 x float>, <4 x float>* %pb, i64 %idx2
  %c = getelementptr <4 x float>, <4 x float>* %pc, i64 %idx2
  %d = getelementptr <4 x float>, <4 x float>* %pd, i64 %idx2
  %la = load <4 x float>, <4 x float>* %a, align 16
  %lb = load <4 x float>, <4 x float>* %b, align 16
  %lc = load <4 x float>, <4 x float>* %c, align 16
  %fma = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %la, <4 x float> %lb, <4 x float> %lc)
  store <4 x float> %fma, <4 x float>* %d, align 16
  ret void
}

declare <4x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>)

; CHECK: define spir_kernel void @__vecz_v8_test_calls(ptr %pa, ptr %pb, ptr %pc, ptr %pd)
; CHECK: entry:

; It checks that the fmuladd intrinsic of <4 x float> gets widened by a factor of 8,
; to produce a PAIR of <16 x float>s.

; It concatenates the 8 x <4 x float> inputs into 2 x <16 x float> values
; CHECK: %[[CA0:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CA1:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CA2:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CA3:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[SA0:.+]] = shufflevector <8 x float> %[[CA0]], <8 x float> %[[CA1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK: %[[SA1:.+]] = shufflevector <8 x float> %[[CA2]], <8 x float> %[[CA3]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

; CHECK: %[[CB0:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CB1:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CB2:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CB3:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[SB0:.+]] = shufflevector <8 x float> %[[CB0]], <8 x float> %[[CB1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK: %[[SB1:.+]] = shufflevector <8 x float> %[[CB2]], <8 x float> %[[CB3]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

; CHECK: %[[CC0:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CC1:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CC2:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CC3:.+]] = shufflevector <4 x float> %{{.+}}, <4 x float> %{{.+}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[SC0:.+]] = shufflevector <8 x float> %[[CC0]], <8 x float> %[[CC1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; CHECK: %[[SC1:.+]] = shufflevector <8 x float> %[[CC2]], <8 x float> %[[CC3]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

; CHECK: %[[FMA0:.+]] = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %[[SA0]], <16 x float> %[[SB0]], <16 x float> %[[SC0]])
; CHECK: %[[FMA1:.+]] = call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %[[SA1]], <16 x float> %[[SB1]], <16 x float> %[[SC1]])

; It splits the 2 x <16 x float> results into 8 <4 x float> values
; CHECK: %[[RES0:.+]] = shufflevector <16 x float> %[[FMA0]], <16 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK: %[[RES1:.+]] = shufflevector <16 x float> %[[FMA0]], <16 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[RES2:.+]] = shufflevector <16 x float> %[[FMA0]], <16 x float> undef, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; CHECK: %[[RES3:.+]] = shufflevector <16 x float> %[[FMA0]], <16 x float> undef, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; CHECK: %[[RES4:.+]] = shufflevector <16 x float> %[[FMA1]], <16 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK: %[[RES5:.+]] = shufflevector <16 x float> %[[FMA1]], <16 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[RES6:.+]] = shufflevector <16 x float> %[[FMA1]], <16 x float> undef, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; CHECK: %[[RES7:.+]] = shufflevector <16 x float> %[[FMA1]], <16 x float> undef, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; CHECK: store <4 x float> %[[RES0]], ptr %{{.+}}, align 16
; CHECK: store <4 x float> %[[RES1]], ptr %{{.+}}, align 16
; CHECK: store <4 x float> %[[RES2]], ptr %{{.+}}, align 16
; CHECK: store <4 x float> %[[RES3]], ptr %{{.+}}, align 16
; CHECK: store <4 x float> %[[RES4]], ptr %{{.+}}, align 16
; CHECK: store <4 x float> %[[RES5]], ptr %{{.+}}, align 16
; CHECK: store <4 x float> %[[RES6]], ptr %{{.+}}, align 16
; CHECK: store <4 x float> %[[RES7]], ptr %{{.+}}, align 16

; CHECK: ret void
