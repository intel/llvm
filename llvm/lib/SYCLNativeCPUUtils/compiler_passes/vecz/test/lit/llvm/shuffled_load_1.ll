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

; RUN: veczc -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @load16(i8 addrspace(1)* %out, i8 addrspace(1)* %in, i32 %stride) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %conv = trunc i64 %call to i32
  %call1 = tail call i64 @__mux_get_global_id(i32 1)
  %conv2 = trunc i64 %call1 to i32
  %mul = mul nsw i32 %conv2, %stride
  %add = add nsw i32 %mul, %conv
  %mul3 = shl nsw i32 %add, 1
  %idx.ext = sext i32 %mul3 to i64
  %add.ptr = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 %idx.ext
  %0 = load i8, i8 addrspace(1)* %add.ptr, align 1
  %arrayidx4 = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr, i64 1
  %1 = load i8, i8 addrspace(1)* %arrayidx4, align 1
  %add7 = add i8 %1, %0
  %idxprom = sext i32 %add to i64
  %arrayidx11 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 %idxprom
  store i8 %add7, i8 addrspace(1)* %arrayidx11, align 1
  ret void
}

declare i64 @__mux_get_global_id(i32)

; CHECK: spir_kernel void @load16
; CHECK: load <4 x i8>
; CHECK: load <4 x i8>
; CHECK-NOT: load <4 x i8>
; CHECK-NOT: call <4 x i8> @__vecz_b_interleaved_load
; CHECK-NOT: call <4 x i8> @__vecz_b_gather_load
; CHECK: shufflevector <4 x i8>
; CHECK: shufflevector <4 x i8>
; CHECK-NOT: shufflevector <4 x i8>
; CHECK: ret void
