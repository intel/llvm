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

; RUN: veczc -k boscc_killer -vecz-passes=vecz-loop-rotate,cfg-convert -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_local_id(i32)
declare i64 @__mux_get_local_size(i32)

@boscc_killer.shared = internal unnamed_addr addrspace(3) global i32 poison, align 4

; Function Attrs: convergent nounwind
define spir_kernel void @boscc_killer(float addrspace(1)* %A, float addrspace(1)* %B, i32 %N, i32 %lda) {
entry:
  %gid0 = tail call i64 @__mux_get_local_id(i32 0)
  %cmp0 = icmp eq i64 %gid0, 0
  br i1 %cmp0, label %if.then, label %if.end

if.then:                                        ; preds = %if.end24
  store i32 %N, i32 addrspace(3)* @boscc_killer.shared, align 4
  br label %if.end

if.end:                                         ; preds = %for.end, %if.end24
  %ldl.a = load i32, i32 addrspace(3)* @boscc_killer.shared, align 4
  %ldl.b = trunc i64 %gid0 to i32
  %ldl = add i32 %ldl.a, %ldl.b
  %cmp1 = icmp eq i32 %ldl, 0
  br i1 %cmp1, label %if.then2, label %if.else

if.else:                                       ; preds = %if.end
  %cmp2 = icmp slt i32 %ldl, %N
  br i1 %cmp2, label %for.body, label %exit

for.body:                                   ; preds = %for.inc, %if.end227
  %acc = phi i32 [ %update2, %for.inc ], [ 1, %if.else ]
  %acc_shl = shl nuw nsw i32 %acc, 2
  %update = add i32 %ldl, %acc_shl
  %cmp3 = icmp slt i32 %update, %ldl
  br i1 %cmp3, label %for.if.then, label %for.inc

for.if.then:                                    ; preds = %for.body
  %mul297.us = mul nsw i32 %update, %lda
  %add298.us = add nsw i32 %mul297.us, %ldl
  %idxprom299.us = sext i32 %add298.us to i64
  %arrayidx300.us = getelementptr inbounds float, float addrspace(1)* %A, i64 %idxprom299.us
  store float zeroinitializer, float addrspace(1)* %arrayidx300.us, align 16
  br label %for.inc

for.inc:                                     ; preds = %for.if.then, %for.body
  %update2 = add nuw nsw i32 %acc, 1
  %cmp4 = icmp ult i32 %acc, 4
  br i1 %cmp4, label %for.body, label %exit

if.then2:                                        ; preds = %if.end
  %gid0_trunc = trunc i64 %gid0 to i32
  %cmp5 = icmp sgt i32 %ldl, %gid0_trunc
  br i1 %cmp5, label %if.then3, label %exit

if.then3:                             ; preds = %for.cond.exit, %if.then53
  %arrayidxB = getelementptr inbounds float, float addrspace(1)* %B, i64 %gid0
  %v23 = load float, float addrspace(1)* %arrayidxB, align 16
  %arrayidxA = getelementptr inbounds float, float addrspace(1)* %A, i64 %gid0
  store float %v23, float addrspace(1)* %arrayidxA, align 16
  %call149 = tail call i64 @__mux_get_local_size(i32 0) #6
  %conv152 = add i64 %call149, %gid0
  %cmp71 = icmp slt i64 %conv152, 0
  br label %exit

exit:                                          ; preds = %for.inc, %if.end227, %for.cond.exit, %if.then53, %entry
  ret void
}

; We mostly want to check that it succeeded since this CFG crashed the block
; ordering algorithm, however it seems it is not easy to create a UnitCL test
; for this, since the CFG gets changed into something that doesn't cause the
; crash. This bug was identified from an Ecosystem failure, however, so it must
; be possible to do somehow.
;
; CHECK: spir_kernel void @__vecz_v4_boscc_killer
; CHECK: entry:
; CHECK: br i1 %{{.+}}, label %if.then.uniform, label %entry.boscc_indir
; CHECK: if.then.uniform:
; CHECK: br label %if.end
; CHECK: entry.boscc_indir:
; CHECK: br i1 %{{.+}}, label %if.end, label %if.then
; CHECK: if.then:
; CHECK: br label %if.end
; CHECK: if.end:
; CHECK: br i1 %{{.+}}, label %if.then2.uniform, label %if.end.boscc_indir
; CHECK: if.else.uniform:
; CHECK: br i1 %{{.+}}, label %for.body.preheader.uniform, label %if.else.uniform.boscc_indir
; CHECK: for.body.preheader.uniform:
; CHECK: br label %for.body.uniform
; CHECK: if.else.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %exit, label %for.body.preheader
; CHECK: for.body.uniform:
; CHECK: br i1 %{{.+}}, label %for.if.then.uniform, label %for.body.uniform.boscc_indir
; CHECK: for.if.then.uniform:
; CHECK: br label %for.inc.uniform
; CHECK: for.body.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %for.inc.uniform, label %for.body.uniform.boscc_store
; CHECK: for.body.uniform.boscc_store:
; CHECK: br label %for.if.then
; CHECK: for.inc.uniform:
; CHECK: br i1 %{{.+}}, label %for.body.uniform, label %exit.loopexit.uniform
; CHECK: exit.loopexit.uniform:
; CHECK: br label %exit
; CHECK: if.then2.uniform:
; CHECK: br i1 %{{.+}}, label %if.then3.uniform, label %if.then2.uniform.boscc_indir
; CHECK: if.end.boscc_indir:
; CHECK: br i1 %{{.+}}, label %if.else.uniform, label %if.else
; CHECK: if.then3.uniform:
; CHECK: br label %exit
; CHECK: if.then2.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %exit, label %if.then3
; CHECK: if.else:
; CHECK: br label %for.body.preheader
; CHECK: for.body.preheader:
; CHECK: br label %for.body
; CHECK: for.body:
; CHECK: br label %for.if.then
; CHECK: for.if.then:
; CHECK: br label %for.inc
; CHECK: for.inc:
; CHECK: br i1 %{{.+}}, label %for.body, label %exit.loopexit
; CHECK: if.then2:
; CHECK: br label %if.then3
; CHECK: if.then3:
; CHECK: br label %exit
; CHECK: exit.loopexit:
; CHECK: br label %if.then2
; CHECK: exit:
; CHECK: ret void
