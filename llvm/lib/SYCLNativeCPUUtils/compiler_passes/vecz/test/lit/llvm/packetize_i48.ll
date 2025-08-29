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

; RUN: veczc -k test -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_local_id(i32)

define spir_kernel void @test(ptr %0, ptr %1) {
entry:
  %lid = tail call i64 @__mux_get_local_id(i32 0)
  %ptr.0 = getelementptr i32, ptr %0, i64 %lid
  %ptr.1 = getelementptr i32, ptr %1, i64 %lid
  %val = load i48, ptr %ptr.0
  store i48 %val, ptr %ptr.1
  ret void
}

; CHECK-LABEL: define spir_kernel void @test
; CHECK: load i48
; CHECK-NOT: load i48
; CHECK: store i48
; CHECK-NOT: store i48

; CHECK-LABEL: define spir_kernel void @__vecz_v4_test
; CHECK: load i48
; CHECK: load i48
; CHECK: load i48
; CHECK: load i48
; CHECK-NOT: load i48
; CHECK: store i48
; CHECK: store i48
; CHECK: store i48
; CHECK: store i48
; CHECK-NOT: store i48
