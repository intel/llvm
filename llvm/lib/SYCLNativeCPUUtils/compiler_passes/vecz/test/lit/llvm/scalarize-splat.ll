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

; RUN: veczc -k splat -vecz-simd-width=4 -vecz-passes=scalarize -vecz-choices=FullScalarization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define dso_local spir_kernel void @splat(float addrspace(1)* %data, float addrspace(1)* %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 noundef 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %data, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %splat.splatinsert = insertelement <4 x float> poison, float %0, i64 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %call1 = tail call spir_func float @not_scalarizable(<4 x float> noundef %splat.splat)
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %call1, float addrspace(1)* %arrayidx2, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32 noundef)
declare spir_func float @not_scalarizable(<4 x float> noundef)

; It checks that the scalarizer turns the original vector splat back into a vector splat,
; instead of a series of insertelement instructions.
; CHECK: void @__vecz_v4_splat({{.*}})
; CHECK: entry:
; CHECK:   %[[LD:.*]] = load float
; CHECK:   %[[INS0:.*]] = insertelement <4 x float> poison, float %[[LD]], {{i32|i64}} 0
; CHECK-NOT: %{{.*}} = insertelement <4 x float> %{{.*}}, float %[[LD]], {{i32|i64}} 1
; CHECK-NOT: %{{.*}} = insertelement <4 x float> %{{.*}}, float %[[LD]], {{i32|i64}} 2
; CHECK-NOT: %{{.*}} = insertelement <4 x float> %{{.*}}, float %[[LD]], {{i32|i64}} 3
; CHECK:   %[[SPLAT:.*]] = shufflevector <4 x float> %[[INS0]], <4 x float> poison, <4 x i32> zeroinitializer
; CHECK:   %{{.*}} = tail call spir_func float @not_scalarizable(<4 x float> noundef %[[SPLAT]])
