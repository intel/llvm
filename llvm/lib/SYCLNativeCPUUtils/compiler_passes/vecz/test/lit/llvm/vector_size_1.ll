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

; RUN: veczc -k test -vecz-simd-width=4 -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr %src, ptr %dst) {
entry:
  %lid = tail call i32 @__mux_get_sub_group_local_id()
  %lid.i64 = zext i32 %lid to i64
  %src.i = getelementptr i64, ptr %src, i64 %lid.i64
  %val = load <1 x i64>, ptr %src.i, align 8
  %vec = shufflevector <1 x i64> %val, <1 x i64> zeroinitializer, <8 x i32> zeroinitializer
  %dst.i = getelementptr <8 x i64>, ptr %dst, i64 %lid.i64
  store <8 x i64> %vec, ptr %dst.i, align 16
  ret void
}

; CHECK-LABEL: define spir_kernel void @test
; CHECK-LABEL: define spir_kernel void @__vecz_v4_test

declare i32 @__mux_get_sub_group_local_id()
