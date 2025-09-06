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

; RUN: veczc -k memcpy_align -vecz-passes=builtin-inlining -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @memcpy_align(ptr align(16) %out, ptr align(8) %in) {
entry:
; CHECK:  %[[A:.*]] = getelementptr inbounds i8, ptr %in, i64 0
; CHECK:  %[[B:.*]] = getelementptr inbounds i8, ptr %out, i64 0
; CHECK:  %[[C:.*]] = load i64, ptr %[[A]], align 8
; CHECK:  store i64 %[[C]], ptr %[[B]], align 16

; CHECK:  %[[D:.*]] = getelementptr inbounds i8, ptr %in, i64 8
; CHECK:  %[[E:.*]] = getelementptr inbounds i8, ptr %out, i64 8
; CHECK:  %[[F:.*]] = load i64, ptr %[[D]], align 8
; CHECK:  store i64 %[[F]], ptr %[[E]], align 8
  call void @llvm.memcpy.p0.p0.i32(ptr noundef align(16) %out, ptr noundef align(8) %in, i32 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
