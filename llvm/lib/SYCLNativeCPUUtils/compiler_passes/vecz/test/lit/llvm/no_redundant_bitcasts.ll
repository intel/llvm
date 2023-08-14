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

; RUN: veczc -k memop_loop_dep -vecz-passes=scalarize -vecz-choices=FullScalarization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-s128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @memop_loop_dep(i32 addrspace(1)* %in, i32 addrspace(1)* %out, i32 %i, i32 %e) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %cmp1 = icmp slt i32 %i, %e
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %i.addr.02 = phi i32 [ %i, %for.body.lr.ph ], [ %inc, %for.inc ]
  %0 = shl i64 %call, 2
  %vload_base = getelementptr i32, i32 addrspace(1)* %in, i64 %0
  %vload_ptr = bitcast i32 addrspace(1)* %vload_base to <4 x i32> addrspace(1)*
  %vload = load <4 x i32>, <4 x i32> addrspace(1)* %vload_ptr, align 16
  %1 = shl i64 %call, 2
  %vstore_base = getelementptr i32, i32 addrspace(1)* %out, i64 %1
  %vstore_ptr = bitcast i32 addrspace(1)* %vstore_base to <4 x i32> addrspace(1)*
  store <4 x i32> %vload, <4 x i32> addrspace(1)* %vstore_ptr, align 16
  %2 = extractelement <4 x i32> %vload, i64 0
  %tobool = icmp ne i32 %2, 0
  %tobool2 = icmp eq i64 %call, 0
  %or.cond = and i1 %tobool2, %tobool
  br i1 %or.cond, label %while.cond.preheader, label %for.inc

while.cond.preheader:                             ; preds = %for.body
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %while.cond.preheader
  %tobool3 = icmp eq i64 %call, 0
  br i1 %tobool3, label %for.inc.loopexit, label %while.cond

for.inc.loopexit:                                 ; preds = %while.cond
  br label %for.inc

for.inc:                                          ; preds = %for.inc.loopexit, %for.body
  %inc = add nsw i32 %i.addr.02, 1
  %exitcond = icmp ne i32 %inc, %e
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

; CA-1431 when we scalarize the vector load, the pointer bitcast back to the
; scalar type is not needed, since the original pointer was the same scalar
; type and can be used directly.

; CHECK: define spir_kernel void @__vecz_v4_memop_loop_dep

; Make sure Scalarization doesn't create any redundant bitcasts
; CHECK-NOT: bitcast
; CHECK: getelementptr i32, ptr addrspace(1) %{{.+}}, i32 0
; CHECK-NOT: bitcast
; CHECK: load i32
; CHECK-NOT: bitcast

; Make sure there is no duplicate GEP that gets the 0-indexed element from the vector
; CHECK-NOT: getelementptr i32, ptr addrspace(1) %{{.+}}, i32 0
; CHECK-NOT: bitcast
; CHECK: load i32
; CHECK-NOT: bitcast
; CHECK: load i32
; CHECK-NOT: bitcast
; CHECK: load i32
; CHECK-NOT: bitcast
