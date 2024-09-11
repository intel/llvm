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

; RUN: veczc -k test_uniform_branch -vecz-passes=packetizer -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_branch(i32 %a, i32* %b) {
entry:
  %conv = sext i32 %a to i64
  %call = call i64 @__mux_get_global_id(i32 0)
  %cmp = icmp eq i64 %conv, %call
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 11, i32* %arrayidx, align 4
  br label %if.end

if.else:
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 42
  store i32 13, i32* %arrayidx2, align 4
  br label %if.end

if.end:
  ret void
}

define spir_kernel void @test_uniform_branch(i32 %a, i32* %b) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %cmp = icmp eq i32 %a, 42
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %idxprom = sext i32 %a to i64
  %idxadd = add i64 %idxprom, %call
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxadd
  store i32 11, i32* %arrayidx, align 4
  br label %if.end

if.else:
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %call
  store i32 13, i32* %arrayidx2, align 4
  br label %if.end

if.end:
  %ptr = phi i32* [ %arrayidx, %if.then ], [ %arrayidx2, %if.else ]
  %ptrplus = getelementptr inbounds i32, i32* %ptr, i64 %call
  store i32 17, i32* %ptrplus, align 4
  ret void
}

define spir_func void @test_nonvarying_loadstore(i32* %a, i32* %b, i32* %c) {
  %index = call i64 @__mux_get_global_id(i32 0)
  %a.i = getelementptr i32, i32* %a, i64 %index
  %b.i = getelementptr i32, i32* %b, i64 %index
  %c.i = getelementptr i32, i32* %c, i64 %index
  %a.load = load i32, i32* %a.i, align 4
  %b.load = load i32, i32* %b.i, align 4
  %add = add i32 %a.load, %b.load
  store i32 %add, i32* %c.i
  ret void
}

declare i64 @__mux_get_global_id(i32)

; This test checks if the if blocks are vectorized without masks and if the phi
; node is also vectorized properly
; CHECK: define spir_kernel void @__vecz_v4_test_uniform_branch(i32 %a, ptr %b)
; CHECK: %call = call i64 @__mux_get_global_id(i32 0)
; CHECK: %[[SPLATINSERT:.+]] = insertelement <4 x i64> {{poison|undef}}, i64 %call, {{i32|i64}} 0
; CHECK: %[[SPLAT:.+]] = shufflevector <4 x i64> %[[SPLATINSERT]], <4 x i64> {{poison|undef}}, <4 x i32> zeroinitializer
; CHECK: %[[GID:.+]] = add <4 x i64> %[[SPLAT]], <i64 0, i64 1, i64 2, i64 3>
; CHECK: %cmp = icmp eq i32 %a, 42
; CHECK: br i1 %cmp, label %if.then, label %if.else

; CHECK: if.then:
; CHECK: %[[GEP1:.+]] = getelementptr i32, ptr %b, <4 x i64>
; CHECK: store <4 x i32> <i32 11, i32 11, i32 11, i32 11>, ptr %{{.+}}, align 4
; CHECK: br label %if.end

; CHECK: if.else:
; CHECK: %[[GEP2:.+]] = getelementptr i32, ptr %b, <4 x i64>
; CHECK: store <4 x i32> <i32 13, i32 13, i32 13, i32 13>, ptr %{{.+}}, align 4
; CHECK: br label %if.end

; CHECK: if.end:
; CHECK: %[[PTR:.+]] = phi <4 x ptr> [ %[[GEP1]], %if.then ], [ %[[GEP2]], %if.else ]
; CHECK: ret void
