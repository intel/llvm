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

define spir_kernel void @remove_intptr(i8 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %0 = ptrtoint i8 addrspace(1)* %in to i64
  %shl = shl nuw nsw i64 %call, 2
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %shl
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %x.07 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %intin.06 = phi i64 [ %0, %entry ], [ %add, %for.body ]
  %add = add i64 %intin.06, 4
  %1 = inttoptr i64 %add to i32 addrspace(1)*
  %2 = load i32, i32 addrspace(1)* %1, align 4
  store i32 %2, i32 addrspace(1)* %arrayidx, align 4
  %inc = add nuw nsw i32 %x.07, 1
  %exitcond.not = icmp eq i32 %inc, 4
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare i64 @__mux_get_global_id(i32)

; CHECK: spir_kernel void @__vecz_v4_remove_intptr
; CHECK-NOT: ptrtoint
; CHECK-NOT: inttoptr
; CHECK: %[[RPHI:.+]] = phi ptr addrspace(1) [ %in, %entry ], [ %[[RGEP:.+]], %for.body ]
; CHECK: %[[RGEP]] = getelementptr i8, ptr addrspace(1) %[[RPHI]], i{{32|64}} 4
; CHECK: load i32, ptr addrspace(1) %[[RGEP]], align 4
