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

declare spir_func i64 @_Z13get_global_idj(i32)

declare i32 @llvm.abs.i32(i32, i1)
declare <2 x i32> @llvm.abs.v2i32(<2 x i32>, i1)

define spir_kernel void @absff(i32* %pa, i32* %pb) {
entry:
  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
  %a = getelementptr i32, i32* %pa, i64 %idx
  %b = getelementptr i32, i32* %pb, i64 %idx
  %la = load i32, i32* %a, align 16
  %res = call spir_func i32 @llvm.abs.i32(i32 %la, i1 true)
  store i32 %res, i32* %b, align 16
  ret void
}

define spir_kernel void @absvf(<2 x i32>* %pa, <2 x i32>* %pb) {
entry:
  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
  %a = getelementptr <2 x i32>, <2 x i32>* %pa, i64 %idx
  %b = getelementptr <2 x i32>, <2 x i32>* %pb, i64 %idx
  %la = load <2 x i32>, <2 x i32>* %a, align 16
  %res = call spir_func <2 x i32> @llvm.abs.v2i32(<2 x i32> %la, i1 true)
  store <2 x i32> %res, <2 x i32>* %b, align 16
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_absff(ptr %pa, ptr %pb)
; CHECK: entry:
; CHECK: %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
; CHECK: %a = getelementptr i32, ptr %pa, i64 %idx
; CHECK: %b = getelementptr i32, ptr %pb, i64 %idx
; CHECK: %[[T0:.*]] = load <4 x i32>, ptr %a, align 4
; CHECK: %[[RES1:.+]] = call <4 x i32> @llvm.abs.v4i32(<4 x i32> %[[T0]], i1 true)
; CHECK: store <4 x i32> %[[RES1]], ptr %b, align 4
; CHECK: ret void

; CHECK: define spir_kernel void @__vecz_v4_absvf(ptr %pa, ptr %pb)
; CHECK: entry:
; CHECK: %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
; CHECK: %a = getelementptr <2 x i32>, ptr %pa, i64 %idx
; CHECK: %b = getelementptr <2 x i32>, ptr %pb, i64 %idx
; CHECK: %[[T0:.*]] = load <8 x i32>, ptr %a, align 4
; CHECK: %[[RES2:.+]] = call <8 x i32> @llvm.abs.v8i32(<8 x i32> %[[T0]], i1 true)
; CHECK: store <8 x i32> %[[RES2]], ptr %b, align 8
; CHECK: ret void
