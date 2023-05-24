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

; This test checks that we create a new preheader that blends the preheader
; of the uniform and the predicated paths for a loop that has not been
; duplicated (because of the barrier in it).

; RUN: %veczc -k duplicate_preheader -vecz-passes="function(instcombine,simplifycfg),mergereturn,vecz-loop-rotate,function(loop(indvars)),cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | %filecheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: noduplicate
declare void @__mux_work_group_barrier(i32, i32, i32) #1
; Function Attrs: nounwind readnone
declare spir_func i64 @_Z12get_local_idj(i32)

define spir_kernel void @duplicate_preheader(i32 addrspace(1)* %out, i32 %n) {
entry:
  %id = tail call spir_func i64 @_Z12get_local_idj(i32 0)
  %cmp = icmp sgt i64 %id, 3
  br i1 %cmp, label %if.then, label %if.end

if.then:                                     ; preds = %entry
  br label %for.cond

for.cond:
  %ret.0 = phi i64 [ 0, %if.then ], [ %inc, %for.body ]
  %storemerge8 = phi i32 [ 0, %if.then ], [ %inc4, %for.body ]
  %mul = shl nsw i32 %n, 1
  %cmp2 = icmp uge i32 %storemerge8, %mul
  br i1 %cmp2, label %for.body, label %if.end

for.body:
  %inc = add nsw i64 %ret.0, 1
  %inc4 = add nsw i32 %storemerge8, 1
  br label %for.cond

if.end:                                     ; preds = %if.then, %entry
  %idx.blend = phi i64 [ %id, %entry ], [ %ret.0, %for.cond ]
  %gep_var = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idx.blend
  br label %barrier

barrier:                                     ; preds = %latch, %if.end
  call void @__mux_work_group_barrier(i32 0, i32 1, i32 272)
  br i1 %cmp, label %body, label %latch

body:                                     ; preds = %barrier
  %gep_uni = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  %ret = load i32, i32 addrspace(1)* %gep_uni, align 16
  store i32 %ret, i32 addrspace(1)* %gep_var, align 16
  br label %latch

latch:                                     ; preds = %body, %barrier
  %cmp3 = icmp sgt i32 %n, 10
  br i1 %cmp3, label %exit, label %barrier

exit:                                     ; preds = %latch
  ret void
}

attributes #1 = { noduplicate }

; CHECK: spir_kernel void @__vecz_v4_duplicate_preheader
; CHECK: br i1 %{{.+}}, label %[[FORCONDPREHEADERUNIFORM:.+]], label %[[ENTRYBOSCCINDIR:.+]]

; Make sure we have both the uniform and non-uniform versions of the for loop.
; CHECK: [[FORCONDPREHEADERUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM:.+]]

; CHECK: [[ENTRYBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFEND:.+]], label %[[FORCONDPREHEADER:.+]]

; CHECK: [[FORCONDUNIFORM]]:
; CHECK: br i1 {{(%([0-9A-Za-z\.])+)|(false)}}, label %[[IFENDLOOPEXITUNIFORM:.+]], label %[[FORBODYUNIFORM:.+]]

; CHECK: [[FORBODYUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM]]

; CHECK: [[IFENDLOOPEXITUNIFORM]]:
; CHECK: br label %[[IFEND]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%([0-9A-Za-z\.])+)|(false)}}, label %[[IFENDLOOPEXIT:.+]], label %[[FORBODY:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[IFENDLOOPEXIT]]:
; CHECK: br label %[[IFEND]]

; Make sure we're reconverging here from the uniform and predicated paths before
; branching to the barrier.
; CHECK: [[IFEND]]:{{.*}}preds
; CHECK-DAG: %[[IFENDLOOPEXIT]]
; CHECK-DAG: %[[IFENDLOOPEXITUNIFORM]]
; CHECK: br label %[[BARRIER:.+]]

; CHECK: [[BARRIER]]:
; CHECK: br i1 %{{.+}}, label %[[BODYUNIFORM:.+]], label %[[BARRIERBOSCCINDIR:.+]]

; CHECK: [[BODYUNIFORM]]:
; CHECK: br label %[[LATCHUNIFORM:.+]]

; CHECK: [[BARRIERBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[LATCH:.+]], label %[[BODY:.+]]

; CHECK: [[BODY]]:
; CHECK: br label %[[LATCH]]

; CHECK: [[LATCH]]:
; CHECK: %[[CMP3:.+]] = icmp
; CHECK: br i1 %[[CMP3]], label %[[EXIT:.+]], label %[[BARRIER]]

; CHECK: [[EXIT]]:
; CHECK: ret void
