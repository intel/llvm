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

; RUN: %veczc -S < %s -w 16 | %filecheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @add(i32 addrspace(1)* %in1, i32 addrspace(1)* %in2, i32 addrspace(1)* %out, i64 addrspace(1)* %N) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %0 = load i64, i64 addrspace(1)* %N, align 8
  %cmp = icmp ult i64 %call, %0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %call
  %1 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %in2, i64 %call
  %2 = load i32, i32 addrspace(1)* %arrayidx1, align 4
  %add = add nsw i32 %2, %1
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %add, i32 addrspace(1)* %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK: spir_kernel void @__vecz_v16_add
; CHECK: entry:
; CHECK: br i1 %{{.+}}, label %[[END:.+]], label %[[THEN:.+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[THEN]]:
; CHECK: br label %[[END]]
; CHECK-EMPTY:
; CHECK-NEXT: [[END]]:
; CHECK-NEXT: ret void
