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

; RUN: not veczc -k noduplicate:4,8 -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @noduplicate(i32 addrspace(1)* %in1, i32 addrspace(1)* %out) {
entry:
  %tid = call i64 @__mux_get_global_id(i32 0) #3
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid
  %i1 = load i32, i32 addrspace(1)* %arrayidx, align 16
  %dec = call i32 @llvm.loop.decrement.reg.i32(i32 %i1, i32 4)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid
  store i32 %dec, i32 addrspace(1)* %arrayidx2, align 16
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare i32 @llvm.loop.decrement.reg.i32(i32, i32)

;CHECK: Failed to vectorize function 'noduplicate'
