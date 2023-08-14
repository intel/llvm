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

; RUN: veczc -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @test(i32 addrspace(1)* %src, i32 addrspace(1)* %dst, i32 %n) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %conv = trunc i64 %call to i32
  %0 = load i32, i32 addrspace(1)* %src, align 4
  %add = add nsw i32 %conv, %n
  %idxprom = sext i32 %add to i64
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %dst, i64 %idxprom
  store i32 %0, i32 addrspace(1)* %arrayidx1, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)

; CHECK: spir_kernel void @test
; CHECK: store <
