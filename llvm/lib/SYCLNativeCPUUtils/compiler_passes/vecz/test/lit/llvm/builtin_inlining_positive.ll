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

; RUN: veczc -k test -vecz-passes=builtin-inlining -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(float %a, float %b, i32* %c) {
entry:
  %gid = call i64 @__mux_get_global_id(i32 0)
  %cmp = call spir_func i32 @_Z9isgreaterff(float %a, float %b)
  %c0 = getelementptr i32, i32* %c, i64 %gid
  store i32 %cmp, i32* %c0, align 4
  %cmp1 = call spir_func i32 @_Z6islessff(float %a, float %b)
  %c1 = getelementptr i32, i32* %c0, i32 1
  store i32 %cmp1, i32* %c1, align 4
  %cmp2 = call spir_func i32 @_Z7isequalff(float %a, float %b)
  %c2 = getelementptr i32, i32* %c0, i32 2
  store i32 %cmp2, i32* %c2, align 4
  %cmp3 = call spir_func i32 @opt_Z7isequalff(float %a, float %b)
  %c3 = getelementptr i32, i32* %c0, i32 3
  store i32 %cmp3, i32* %c3, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare spir_func i32 @_Z9isgreaterff(float, float)
declare spir_func i32 @_Z6islessff(float, float)
declare spir_func i32 @_Z7isequalff(float, float)

; Test that a non-builtin function is inlined.
define spir_func i32 @opt_Z7isequalff(float, float) {
  ret i32 zeroinitializer
}

; CHECK: define spir_kernel void @__vecz_v4_test(float %a, float %b, ptr %c)
; CHECK: entry:
; CHECK: %gid = call i64 @__mux_get_global_id(i32 0)
; CHECK: %relational = fcmp ogt float %a, %b
; CHECK: %relational[[R1:[0-9]+]] = zext i1 %relational to i32
; CHECK: %c0 = getelementptr i32, ptr %c, i64 %gid
; CHECK: store i32 %relational[[R1]], ptr %c0, align 4
; CHECK: %relational[[R2:[0-9]+]] = fcmp olt float %a, %b
; CHECK: %relational[[R3:[0-9]+]] = zext i1 %relational[[R2:[0-9]+]] to i32
; CHECK: %c1 = getelementptr i32, ptr %c0, {{(i32|i64)}} 1
; CHECK: store i32 %relational[[R3:[0-9]+]], ptr %c1, align 4
; CHECK: %relational[[R4:[0-9]+]] = fcmp oeq float %a, %b
; CHECK: %relational[[R5:[0-9]+]] = zext i1 %relational[[R4:[0-9]+]] to i32
; CHECK: %c2 = getelementptr i32, ptr %c0, {{(i32|i64)}} 2
; CHECK: store i32 %relational[[R5:[0-9]+]], ptr %c2, align 4
; CHECK: %c3 = getelementptr i32, ptr %c0, {{(i32|i64)}} 3
; CHECK: store i32 0, ptr %c3, align 4
; CHECK: ret void
