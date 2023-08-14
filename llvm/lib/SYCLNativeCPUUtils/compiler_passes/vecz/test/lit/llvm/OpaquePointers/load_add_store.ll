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

; RUN: veczc -vecz-simd-width=4 -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @load_add_store(ptr %aptr, ptr %bptr, ptr %zptr) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxa = getelementptr inbounds i32, ptr %aptr, i64 %idx
  %arrayidxb = getelementptr inbounds i32, ptr %bptr, i64 %idx
  %arrayidxz = getelementptr inbounds i32, ptr %zptr, i64 %idx
  %a = load i32, ptr %arrayidxa, align 4
  %b = load i32, ptr %arrayidxb, align 4
  %sum = add i32 %a, %b
  store i32 %sum, ptr %arrayidxz, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_load_add_store(ptr %aptr, ptr %bptr, ptr %zptr)
; CHECK: %idx = call i64 @__mux_get_global_id(i32 0)
; CHECK: %arrayidxa = getelementptr inbounds i32, ptr %aptr, i64 %idx
; CHECK: %arrayidxb = getelementptr inbounds i32, ptr %bptr, i64 %idx
; CHECK: %arrayidxz = getelementptr inbounds i32, ptr %zptr, i64 %idx
; CHECK: %[[TMP0:.*]] = load <4 x i32>, ptr %arrayidxa, align 4
; CHECK: %[[TMP1:.*]] = load <4 x i32>, ptr %arrayidxb, align 4
; CHECK: %sum1 = add <4 x i32> %[[TMP0]], %[[TMP1]]
; CHECK: store <4 x i32> %sum1, ptr %arrayidxz, align 4
; CHECK: ret void
}
