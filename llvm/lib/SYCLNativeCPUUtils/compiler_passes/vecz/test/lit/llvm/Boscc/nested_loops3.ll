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

; RUN: veczc -k nested_loops3 -vecz-passes="function(instcombine,simplifycfg),mergereturn,vecz-loop-rotate,function(loop(indvars)),cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32) #0

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: convergent nounwind
define spir_kernel void @nested_loops3(float addrspace(1)* %symmat, float addrspace(1)* %data, i32 %m, i32 %n) #2 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #3
  %conv = trunc i64 %call to i32
  %sub = add nsw i32 %m, -1
  %cmp = icmp sgt i32 %sub, %conv
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %mul = mul nsw i32 %conv, %m
  %add = add nsw i32 %mul, %conv
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %symmat, i64 %idxprom
  store float 1.000000e+00, float addrspace(1)* %arrayidx, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.end, %if.then
  %storemerge.in = phi i32 [ %conv, %if.then ], [ %storemerge, %for.end ]
  %storemerge = add nsw i32 %storemerge.in, 1
  %cmp3 = icmp slt i32 %storemerge, %m
  br i1 %cmp3, label %for.cond5, label %if.end

for.cond5:                                        ; preds = %for.body8, %for.cond
  %storemerge1 = phi i32 [ %inc, %for.body8 ], [ 0, %for.cond ]
  %cmp6 = icmp slt i32 %storemerge1, %n
  br i1 %cmp6, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond5
  %mul9 = mul nsw i32 %storemerge1, %m
  %add10 = add nsw i32 %mul9, %conv
  %idxprom11 = sext i32 %add10 to i64
  %arrayidx12 = getelementptr inbounds float, float addrspace(1)* %data, i64 %idxprom11
  %0 = load float, float addrspace(1)* %arrayidx12, align 4
  %mul13 = mul nsw i32 %storemerge1, %m
  %add14 = add nsw i32 %mul13, %storemerge
  %idxprom15 = sext i32 %add14 to i64
  %arrayidx16 = getelementptr inbounds float, float addrspace(1)* %data, i64 %idxprom15
  %1 = load float, float addrspace(1)* %arrayidx16, align 4
  %mul18 = mul nsw i32 %conv, %m
  %add19 = add nsw i32 %storemerge, %mul18
  %idxprom20 = sext i32 %add19 to i64
  %arrayidx21 = getelementptr inbounds float, float addrspace(1)* %symmat, i64 %idxprom20
  %2 = load float, float addrspace(1)* %arrayidx21, align 4
  %3 = call float @llvm.fmuladd.f32(float %0, float %1, float %2)
  store float %3, float addrspace(1)* %arrayidx21, align 4
  %inc = add nuw nsw i32 %storemerge1, 1
  br label %for.cond5

for.end:                                          ; preds = %for.cond5
  %mul22 = mul nsw i32 %conv, %m
  %add23 = add nsw i32 %storemerge, %mul22
  %idxprom24 = sext i32 %add23 to i64
  %arrayidx25 = getelementptr inbounds float, float addrspace(1)* %symmat, i64 %idxprom24
  %4 = load float, float addrspace(1)* %arrayidx25, align 4
  %mul26 = mul nsw i32 %storemerge, %m
  %add27 = add nsw i32 %mul26, %conv
  %idxprom28 = sext i32 %add27 to i64
  %arrayidx29 = getelementptr inbounds float, float addrspace(1)* %symmat, i64 %idxprom28
  store float %4, float addrspace(1)* %arrayidx29, align 4
  br label %for.cond

if.end:                                           ; preds = %for.cond, %entry
  ret void
}

; The purpose of this test is to make sure we correctly set the incoming value
; of a boscc_blend instruction (in a loop header) from the latch as being the
; value defined in the latch iteself.

; CHECK: spir_kernel void @__vecz_v4_nested_loops3
; CHECK: entry:
; CHECK: br i1 %{{.+}}, label %if.then.uniform, label %entry.boscc_indir

; CHECK: if.then.uniform:
; CHECK: br i1 %{{.+}}, label %for.cond5.preheader.lr.ph.uniform, label %if.then.uniform.boscc_indir

; CHECK: entry.boscc_indir:
; CHECK: br i1 %{{.+}}, label %if.end, label %if.then

; CHECK: for.cond5.preheader.lr.ph.uniform:
; CHECK: br label %for.cond5.preheader.uniform

; CHECK: if.then.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %if.end.loopexit.uniform, label %for.cond5.preheader.lr.ph

; CHECK: for.cond5.preheader.uniform:
; CHECK: br label %for.cond5.uniform

; CHECK: for.end.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %for.cond.if.end.loopexit_crit_edge.uniform, label %for.end.uniform.boscc_store

; CHECK: for.end.uniform.boscc_store:
; CHECK: br label %for.cond5.preheader

; CHECK: if.then:
; CHECK: br label %for.cond5.preheader.lr.ph

; CHECK: for.cond5.preheader.lr.ph:
; CHECK: br label %for.cond5.preheader

; CHECK: for.cond5.preheader:

; This is the important bit of the test
; Note that the LCSSA PHI node got cleaned up!
; For some reason LIT needs these checks to be split across two lines
; CHECK: %[[LATCH_VALUE1:.*\.boscc_blend[0-9]*]] = phi i{{32|64}} [ %{{.+}}, %for.end.uniform.boscc_store ],
; CHECK-SAME: [ %[[LATCH_VALUE1]], %for.end ], [ %{{.+}}, %for.cond5.preheader.lr.ph ]

; CHECK: %[[LATCH_VALUE2:.*\.boscc_blend[0-9]*]] = phi i{{32|64}} [ %{{.+}}, %for.end.uniform.boscc_store ],
; CHECK-SAME: [ %[[LATCH_VALUE2]], %for.end ], [ %{{.+}}, %for.cond5.preheader.lr.ph ]

; CHECK: %[[LATCH_VALUE3:.*\.boscc_blend[0-9]*]] = phi i{{32|64}} [ %{{.+}}, %for.end.uniform.boscc_store ],
; CHECK-SAME: [ %[[LATCH_VALUE3]], %for.end ], [ %{{.+}}, %for.cond5.preheader.lr.ph ]

; CHECK: %[[LATCH_VALUE4:.*\.boscc_blend[0-9]*]] = phi i{{32|64}} [ %{{.+}}, %for.end.uniform.boscc_store ],
; CHECK-SAME: [ %[[LATCH_VALUE4]], %for.end ], [ %{{.+}}, %for.cond5.preheader.lr.ph ]

; CHECK: %[[LATCH_VALUE5:.+\.boscc_blend[0-9]*]] = phi i1 [ true, %for.end.uniform.boscc_store ],
; CHECK-SAME: [ %[[LATCH_VALUE5]], %for.end ], [ %{{.+}}, %for.cond5.preheader.lr.ph ]
