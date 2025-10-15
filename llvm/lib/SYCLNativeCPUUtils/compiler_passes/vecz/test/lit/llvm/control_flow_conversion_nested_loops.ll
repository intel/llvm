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

; RUN: veczc -k test_nested_loops -vecz-passes=cfg-convert -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_uniform_if(i32 %a, i32* %b) {
entry:
  %cmp = icmp eq i32 %a, 1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 11, i32* %arrayidx, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds i32, i32* %b, i64 42
  store i32 13, i32* %arrayidx1, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define spir_kernel void @test_varying_if(i32 %a, i32* %b) {
entry:
  %conv = sext i32 %a to i64
  %call = call i64 @__mux_get_global_id(i32 0)
  %cmp = icmp eq i64 %conv, %call
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 11, i32* %arrayidx, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 42
  store i32 13, i32* %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define spir_kernel void @test_uniform_loop(i32 %a, i32* %b)  {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %conv = trunc i64 %call to i32
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %storemerge = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %storemerge, 16
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %storemerge, %a
  %add2 = add nsw i32 %storemerge, %conv
  %idxprom = sext i32 %add2 to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 %add, i32* %arrayidx, align 4
  %inc = add nsw i32 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define spir_kernel void @test_varying_loop(i32 %a, i32* %b) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %conv = trunc i64 %call to i32
  %sub = sub nsw i32 16, %conv
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %storemerge = phi i32 [ %sub, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %storemerge, 16
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %storemerge, %a
  %add2 = add nsw i32 %storemerge, %conv
  %idxprom = sext i32 %add2 to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 %add, i32* %arrayidx, align 4
  %inc = add nsw i32 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define spir_kernel void @test_nested_loops(i32* %a, i32* %b)  {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %conv = trunc i64 %call to i32
  %sub = sub nsw i32 16, %conv
  br label %for.cond

for.cond:                                         ; preds = %for.inc12, %entry
  %storemerge = phi i32 [ %sub, %entry ], [ %inc13, %for.inc12 ]
  %cmp = icmp slt i32 %storemerge, 16
  br i1 %cmp, label %for.body, label %for.end14

for.body:                                         ; preds = %for.cond
  %sub2 = sub nsw i32 24, %conv
  br label %for.cond3

for.cond3:                                        ; preds = %for.body6, %for.body
  %storemerge1 = phi i32 [ %sub2, %for.body ], [ %inc, %for.body6 ]
  %cmp4 = icmp slt i32 %storemerge, 24
  br i1 %cmp4, label %for.body6, label %for.inc12

for.body6:                                        ; preds = %for.cond3
  %add = add nsw i32 %storemerge1, %conv
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add7 = add i32 %storemerge1, %storemerge
  %add8 = add i32 %add7, %0
  %add9 = add nsw i32 %storemerge, %conv
  %idxprom10 = sext i32 %add9 to i64
  %arrayidx11 = getelementptr inbounds i32, i32* %b, i64 %idxprom10
  store i32 %add8, i32* %arrayidx11, align 4
  %inc = add nsw i32 %storemerge1, 1
  br label %for.cond3

for.inc12:                                        ; preds = %for.cond3
  %inc13 = add nsw i32 %storemerge, 1
  br label %for.cond

for.end14:                                        ; preds = %for.cond
  ret void
}

declare i64 @__mux_get_global_id(i32)

; A nested loop, in the form of
;
;  int gid = get_global_id(0);
;  for (int i = 16 - gid; i < 16; ++i) {
;    for (int j = 24 - gid; i < 24; ++j) {
;      b[i + gid] = a[j + gid] + i + j;
;    }
;  }
;
; The important bit is that both of the loops have their iterations dependent on
; the global ID
; CHECK: define spir_kernel void @__vecz_v4_test_nested_loops(ptr %a, ptr %b)
; CHECK: entry:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: %[[ENTRYMASK_FORCOND:.+]] = phi i1 [ true, %entry ], [ %[[FORINC12EXITMASK3:.+]], %[[FORINC12:.+]] ]
; CHECK: %[[EXITMASK1:.+]] = phi i1 [ false, %entry ], [ %[[LOOPEXITMASK2:.+]], %[[FORINC12]] ]
; CHECK: %[[CMP:.+]] = icmp slt i32 %[[STOREMERGE:.+]], 16
; CHECK: %[[EDGEMASK_FORBODY:.+]] = select i1 %[[ENTRYMASK_FORCOND]], i1 %[[CMP]], i1 false
; CHECK: %[[NOT_CMP:.+]] = xor i1 %[[CMP]], true
; CHECK: %[[EDGEMASK_FOREND14:.+]] = select i1 %[[ENTRYMASK_FORCOND]], i1 %[[NOT_CMP]], i1 false
; CHECK: %[[LOOPEXITMASK2]] = or i1 %[[EXITMASK1]], %[[EDGEMASK_FOREND14]]
; CHECK: br label %[[FORBODY:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND3:.+]]

; CHECK: [[FORCOND3]]:
; CHECK: %[[ENTRYMASK_FORCOND3:.+]] = phi i1 [ %[[EDGEMASK_FORBODY:.+]], %[[FORBODY]] ], [ %[[FORBODY6EXITMASK:.+]], %[[FORBODY6:.+]] ]
; CHECK: %[[PREVEXITMASK:.+]] = phi i1 [ false, %[[FORBODY]] ], [ %[[FORINC12LOOPEXITMASKUPDATE:.+]], %[[FORBODY6]] ]
; CHECK: %[[CMP4:.+]] = icmp slt i32 %[[STOREMERGE]], 24
; CHECK: %[[EDGEMASK_FORBODY6:.+]] = select i1 %[[ENTRYMASK_FORCOND3]], i1 %[[CMP4]], i1 false
; CHECK: %[[NOT_CMP4:.+]] = xor i1 %[[CMP4]], true
; CHECK: %[[EDGEMASK_FORINC12:.+]] = select i1 %[[ENTRYMASK_FORCOND3]], i1 %[[NOT_CMP4]], i1 false
; CHECK: %[[FORINC12LOOPEXITMASKUPDATE]] = or i1 %[[PREVEXITMASK]], %[[EDGEMASK_FORINC12]]
; CHECK: br label %[[FORBODY6:.+]]

; CHECK: [[FORBODY6]]:
; CHECK: %[[MGL:.+]] = call i32 @__vecz_b_masked_load4_ju3ptrb(ptr %{{.+}}, i1 %[[EDGEMASK_FORBODY6]])
; CHECK: %[[ADD8:.+]] = add i32 %{{.+}}, %[[MGL]]
; CHECK: call void @__vecz_b_masked_store4_ju3ptrb(i32 %[[ADD8]], ptr %{{.+}}, i1 %[[EDGEMASK_FORBODY6]])
; CHECK: %[[FORBODY6EXITMASK_ANY:.+]] = call i1 @__vecz_b_divergence_any(i1 %[[FORBODY6EXITMASK]])
; CHECK: br i1 %[[FORBODY6EXITMASK_ANY]], label %[[FORCOND3:.+]], label %[[FORINC12:.+]]

; CHECK: [[FORINC12]]:
; CHECK: %[[FORINC12LOOPEXITMASKUPDATE_ANY:.+]] = call i1 @__vecz_b_divergence_any(i1 %[[FORINC12LOOPEXITMASKUPDATE]])
; CHECK: br i1 %[[FORINC12LOOPEXITMASKUPDATE_ANY]], label %[[FORCOND:.+]], label %[[FOREND14:.+]]

; CHECK: [[FOREND14]]:
; CHECK: ret void
