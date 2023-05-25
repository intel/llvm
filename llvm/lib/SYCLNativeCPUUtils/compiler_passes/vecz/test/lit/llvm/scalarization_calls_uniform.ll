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

; RUN: veczc -k test_calls -vecz-passes=scalarize -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_calls(<4 x float>* %a, <4 x float>* %b, <4 x i32>* %c, <4 x float>* %d) {
entry:
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

; Checks that this function gets vectorized, although because every instruction is
; uniform, the process of vectorization makes no actual changes whatsoever!
; CHECK: define spir_kernel void @__vecz_v4_test_calls(ptr %a, ptr %b, ptr %c, ptr %d)
; CHECK: entry:
; CHECK: %[[LA:.+]] = load <4 x float>, ptr %a, align 16
; CHECK: %[[LB:.+]] = load <4 x float>, ptr %b, align 16
; CHECK: %[[LC:.+]] = load <4 x i32>, ptr %c, align 16
; CHECK: %[[CALL:.+]] = call spir_func <4 x float> @_Z14convert_float4Dv4_i(<4 x i32> %[[LC]])
; CHECK: %[[FMAD:.+]] = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %[[LA]], <4 x float> %[[LB]], <4 x float> %[[CALL]])
; CHECK: store <4 x float> %[[FMAD]], ptr %d, align 16
; CHECK: ret void
