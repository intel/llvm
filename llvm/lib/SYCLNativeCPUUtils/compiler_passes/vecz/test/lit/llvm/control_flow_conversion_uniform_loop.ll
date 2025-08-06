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

; RUN: veczc -k test_uniform_loop -vecz-passes=cfg-convert -vecz-simd-width=4 -S < %s | FileCheck %s

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

; This tests for a uniform loop that should remain untouched by the CFC pass
; CHECK: define spir_kernel void @__vecz_v4_test_uniform_loop(i32 %a, ptr %b)
; CHECK: br label %for.cond

; CHECK: for.cond:
; CHECK: %storemerge = phi i32 [ 0, %entry ], [ %inc, %for.body ]
; CHECK: %cmp = icmp slt i32 %storemerge, 16
; CHECK: br i1 %cmp, label %for.body, label %for.end

; CHECK: for.body:
; CHECK: %add = add nsw i32 %storemerge, %a
; CHECK: %idxprom = sext i32 %add2 to i64
; CHECK: %arrayidx = getelementptr i32, ptr %b, i64 %idxprom
; CHECK: store i32 %add, ptr %arrayidx, align 4
; CHECK: %inc = add nsw i32 %storemerge, 1
; CHECK: br label %for.cond

; CHECK: for.end:
; CHECK: ret void
