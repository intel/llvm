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


; RUN: veczc -k regression_by_all -vecz-passes=vecz-loop-rotate,cfg-convert -S < %s | FileCheck %s

; The purpose of this test is to make sure the block `c` does not get considered
; as a by_all because one of its predecessors is by_all. In fact, because `c`
; has a div causing block (b) as one of its predecessors, then it cannot be
; considered by_all

; The CFG of the following kernel is:
;
;   a
;   |\
;   | b
;   |/ \
;   c   d
;    \ /
;     e
;
; * where node a is a uniform branch, and node b is a varying branch.
; * where nodes c, d and e are divergent.
;
; With partial linearization we will have a CFG of the form:
;
;     a
;    /|
;   | b
;   | |
;   | d
;    \|
;     c
;     |
;     e
;
; __kernel void regression_by_all(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   if (n % 2 == 0) {
;     goto d;
;   } else {
;     ret = 1;
;     if (id % 2 != 0) {
;       goto d;
;     } else {
;       for (int i = 0; i < n; ++i) { ret++; }
;       goto e;
;     }
;   }
;
; d:
;   ret += id;
;   ret *= n;
;
; e:
;   out[id] = ret;
; }

; ModuleID = 'kernel.opencl'
source_filename = "kernel.opencl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @regression_by_all(i32 addrspace(1)* %out, i32 %n) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %conv = trunc i64 %call to i32
  %rem1 = and i32 %n, 1
  %cmp = icmp eq i32 %rem1, 0
  br i1 %cmp, label %d, label %if.else

if.else:                                          ; preds = %entry
  %rem22 = and i32 %conv, 1
  %cmp3 = icmp eq i32 %rem22, 0
  br i1 %cmp3, label %for.cond, label %d

for.cond:                                         ; preds = %if.else, %for.body
  %ret.0 = phi i32 [ %inc, %for.body ], [ 1, %if.else ]
  %storemerge = phi i32 [ %inc9, %for.body ], [ 0, %if.else ]
  %cmp7 = icmp slt i32 %storemerge, %n
  br i1 %cmp7, label %for.body, label %e

for.body:                                         ; preds = %for.cond
  %inc = add nuw nsw i32 %ret.0, 1
  %inc9 = add nuw nsw i32 %storemerge, 1
  br label %for.cond

d:                                                ; preds = %if.else, %entry
  %ret.1 = phi i32 [ 0, %entry ], [ 1, %if.else ]
  %add = add nsw i32 %ret.1, %conv
  %mul = mul nsw i32 %add, %n
  br label %e

e:                                                ; preds = %for.cond, %d
  %ret.2 = phi i32 [ %mul, %d ], [ %ret.0, %for.cond ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.2, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: convergent nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK: spir_kernel void @__vecz_v4_regression_by_all
; CHECK: br i1 %[[CMP:.+]], label %[[D:.+]], label %[[IFELSE:.+]]

; CHECK: [[D]]:
; CHECK-NOT: %d.entry_mask = and i1 true, true
; CHECK: %d.entry_mask = phi i1
