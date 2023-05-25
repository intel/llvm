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

; RUN: veczc -k test_calls -vecz-passes=scalarize -vecz-simd-width=4 -vecz-choices=FullScalarization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @test_calls(<4 x float>* %pa, <4 x float>* %pb, <4 x i32>* %pc, <4 x float>* %pd) {
entry:
  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
  %a = getelementptr <4 x float>, <4 x float>* %pa, i64 %idx
  %b = getelementptr <4 x float>, <4 x float>* %pb, i64 %idx
  %c = getelementptr <4 x i32>, <4 x i32>* %pc, i64 %idx
  %d = getelementptr <4 x float>, <4 x float>* %pd, i64 %idx
  %0 = load <4 x float>, <4 x float>* %a, align 16
  %1 = load <4 x float>, <4 x float>* %b, align 16
  %2 = load <4 x i32>, <4 x i32>* %c, align 16
  %call = call spir_func <4 x float> @_Z14convert_float4Dv4_i(<4 x i32> %2)
  %3 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %0, <4 x float> %1, <4 x float> %call)
  store <4 x float> %3, <4 x float>* %d, align 16
  ret void
}

declare spir_func <4 x float> @_Z14convert_float4Dv4_i(<4 x i32>)
declare spir_func float @_Z13convert_floati(i32)
declare <4x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>)

; CHECK: define spir_kernel void @__vecz_v4_test_calls(ptr %pa, ptr %pb, ptr %pc, ptr %pd)
; CHECK: entry:
; CHECK: %[[A_0:.+]] = getelementptr float, ptr %a, i32 0
; CHECK: %[[A_1:.+]] = getelementptr float, ptr %a, i32 1
; CHECK: %[[A_2:.+]] = getelementptr float, ptr %a, i32 2
; CHECK: %[[A_3:.+]] = getelementptr float, ptr %a, i32 3
; CHECK: %[[LA_0:.+]] = load float, ptr %[[A_0]]
; CHECK: %[[LA_1:.+]] = load float, ptr %[[A_1]]
; CHECK: %[[LA_2:.+]] = load float, ptr %[[A_2]]
; CHECK: %[[LA_3:.+]] = load float, ptr %[[A_3]]
; CHECK: %[[B_0:.+]] = getelementptr float, ptr %b, i32 0
; CHECK: %[[B_1:.+]] = getelementptr float, ptr %b, i32 1
; CHECK: %[[B_2:.+]] = getelementptr float, ptr %b, i32 2
; CHECK: %[[B_3:.+]] = getelementptr float, ptr %b, i32 3
; CHECK: %[[LB_0:.+]] = load float, ptr %[[B_0]]
; CHECK: %[[LB_1:.+]] = load float, ptr %[[B_1]]
; CHECK: %[[LB_2:.+]] = load float, ptr %[[B_2]]
; CHECK: %[[LB_3:.+]] = load float, ptr %[[B_3]]
; CHECK: %[[C_0:.+]] = getelementptr i32, ptr %c, i32 0
; CHECK: %[[C_1:.+]] = getelementptr i32, ptr %c, i32 1
; CHECK: %[[C_2:.+]] = getelementptr i32, ptr %c, i32 2
; CHECK: %[[C_3:.+]] = getelementptr i32, ptr %c, i32 3
; CHECK: %[[LC_0:.+]] = load i32, ptr %[[C_0]]
; CHECK: %[[LC_1:.+]] = load i32, ptr %[[C_1]]
; CHECK: %[[LC_2:.+]] = load i32, ptr %[[C_2]]
; CHECK: %[[LC_3:.+]] = load i32, ptr %[[C_3]]
; CHECK: %[[CALL1:.+]] = call spir_func float @_Z13convert_floati(i32 %[[LC_0]])
; CHECK: %[[CALL2:.+]] = call spir_func float @_Z13convert_floati(i32 %[[LC_1]])
; CHECK: %[[CALL3:.+]] = call spir_func float @_Z13convert_floati(i32 %[[LC_2]])
; CHECK: %[[CALL4:.+]] = call spir_func float @_Z13convert_floati(i32 %[[LC_3]])
; CHECK: %[[FMAD_0:.+]] = call float @llvm.fmuladd.f32(float %[[LA_0]], float %[[LB_0]], float %[[CALL1]])
; CHECK: %[[FMAD_1:.+]] = call float @llvm.fmuladd.f32(float %[[LA_1]], float %[[LB_1]], float %[[CALL2]])
; CHECK: %[[FMAD_2:.+]] = call float @llvm.fmuladd.f32(float %[[LA_2]], float %[[LB_2]], float %[[CALL3]])
; CHECK: %[[FMAD_3:.+]] = call float @llvm.fmuladd.f32(float %[[LA_3]], float %[[LB_3]], float %[[CALL4]])
; CHECK: %[[D_0:.+]] = getelementptr float, ptr %d, i32 0
; CHECK: %[[D_1:.+]] = getelementptr float, ptr %d, i32 1
; CHECK: %[[D_2:.+]] = getelementptr float, ptr %d, i32 2
; CHECK: %[[D_3:.+]] = getelementptr float, ptr %d, i32 3
; CHECK: store float %[[FMAD_0]], ptr %[[D_0]]
; CHECK: store float %[[FMAD_1]], ptr %[[D_1]]
; CHECK: store float %[[FMAD_2]], ptr %[[D_2]]
; CHECK: store float %[[FMAD_3]], ptr %[[D_3]]
; CHECK: ret void
