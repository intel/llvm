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
; RUN: veczc -k bar -vecz-simd-width=4 -S -o - %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @__mux_get_global_id(i32)

define void @bar(i64** %ptrptrs, i64 %val) {
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxa = getelementptr inbounds i64*, i64** %ptrptrs, i64 %idx
  %ptrs = load i64*, i64** %arrayidxa, align 4
  %addr = getelementptr inbounds i64, i64* %ptrs, <4 x i32> <i32 2, i32 2, i32 2, i32 2>

  %elt0 = extractelement <4 x i64*> %addr, i32 0
  %elt1 = extractelement <4 x i64*> %addr, i32 1
  %elt2 = extractelement <4 x i64*> %addr, i32 2
  %elt3 = extractelement <4 x i64*> %addr, i32 3

  store i64 %val, i64* %elt0
  store i64 %val, i64* %elt1
  store i64 %val, i64* %elt2
  store i64 %val, i64* %elt3
  ret void
}

; it checks that the GEP with mixed scalar/vector operands in the kernel
; gets scalarized/re-packetized correctly

; CHECK: define void @__vecz_v4_bar
; CHECK: %[[ADDR:.+]] = getelementptr {{i64|i8}}, <4 x ptr> %{{.+}}, {{i64 2|i64 16}}
; CHECK: call void @__vecz_b_scatter_store8_Dv4_mDv4_u3ptr(<4 x i64> %.splat{{.*}}, <4 x ptr> %[[ADDR]])
