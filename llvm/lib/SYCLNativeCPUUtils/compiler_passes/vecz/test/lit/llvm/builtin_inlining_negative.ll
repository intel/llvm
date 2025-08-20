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

; RUN: veczc -k test_rhadd -vecz-passes=builtin-inlining -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_normalize(float %a, float %b, i32* %c) {
entry:
  %gid = call i64 @__mux_get_global_id(i32 0)
  %norm = call spir_func float @_Z9normalizef(float %a)
  %normi = fptosi float %norm to i32
  %c0 = getelementptr i32, i32* %c, i64 %gid
  store i32 %normi, i32* %c0, align 4
  ret void
}

define spir_kernel void @test_rhadd(i32 %a, i32 %b, i32* %c) {
entry:
  %gid = call i64 @__mux_get_global_id(i32 0)
  %add = call spir_func i32 @_Z5rhaddjj(i32 %a, i32 %b)
  %c0 = getelementptr i32, i32* %c, i64 %gid
  store i32 %add, i32* %c0, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare spir_func float @_Z9normalizef(float)
declare spir_func i32 @_Z5rhaddjj(i32, i32)

; CHECK-NOT: define spir_kernel void @__vecz_v4_test_normalize(float %a, float %b, ptr %c)

; CHECK: define spir_kernel void @__vecz_v4_test_rhadd(i32 %a, i32 %b, ptr %c)
; CHECK: entry:
; CHECK: %gid = call i64 @__mux_get_global_id(i32 0)
; CHECK: %add = call spir_func i32 @_Z5rhaddjj(i32 %a, i32 %b)
; CHECK: %c0 = getelementptr i32, ptr %c, i64 %gid
; CHECK: store i32 %add, ptr %c0, align 4
; CHECK: ret void
