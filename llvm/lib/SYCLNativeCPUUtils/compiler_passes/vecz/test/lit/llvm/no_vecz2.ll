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

; RUN: veczc -S < %s -vecz-auto | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @no_vecz2(i32 addrspace(1)* %out, i32 %n, i32 addrspace(1)* %m) {
entry:
  %0 = load i32, i32 addrspace(1)* %m, align 4
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %cmp = icmp eq i64 %call, 0
  br i1 %cmp, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %entry
  %cmp167 = icmp sgt i32 %n, 0
  br i1 %cmp167, label %for.body29.lr.ph, label %for.cond.cleanup28

for.body29.lr.ph:                                 ; preds = %for.cond.preheader
  %add = add i32 %0, 1
  %factor = shl i32 %0, 2
  %1 = shl i32 %n, 2
  %2 = add i32 %1, -4
  %reass.mul = mul i32 %2, %add
  %3 = add i32 %factor, 4
  %4 = add i32 %3, %reass.mul
  br label %for.cond.cleanup28

for.cond.cleanup28:                               ; preds = %for.body29.lr.ph, %for.cond.preheader
  %ret.3.lcssa = phi i32 [ %4, %for.body29.lr.ph ], [ 0, %for.cond.preheader ]
  store i32 %ret.3.lcssa, i32 addrspace(1)* %out, align 4
  br label %if.end

if.end:                                           ; preds = %for.cond.cleanup28, %entry
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK: spir_kernel void @{{(__vecz_v16_)?}}no_vecz2
; CHECK-NOT: extractelement
; CHECK-NOT: define void @__vecz_b_masked_store
; CHECK: store i32
