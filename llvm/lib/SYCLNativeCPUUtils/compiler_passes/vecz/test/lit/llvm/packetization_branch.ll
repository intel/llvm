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

; RUN: veczc -k test_branch -vecz-passes=cfg-convert,packetizer -vecz-simd-width=4 -S < %s | FileCheck %s

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

declare i64 @__mux_get_global_id(i32)

; This test checks if the branch conditions and the branch BBs are vectorized
; and masked properly
; CHECK: define spir_kernel void @__vecz_v4_test_branch(i32 %a, ptr %b)
; CHECK: %conv = sext i32 %a to i64
; CHECK: %[[A_SPLATINSERT:.+]] = insertelement <4 x i64> {{poison|undef}}, i64 %conv, {{i32|i64}} 0
; CHECK: %[[A_SPLAT:.+]] = shufflevector <4 x i64> %[[A_SPLATINSERT]], <4 x i64> {{poison|undef}}, <4 x i32> zeroinitializer
; CHECK: %call = call i64 @__mux_get_global_id(i32 0)
; CHECK: %[[GID_SPLATINSERT:.+]] = insertelement <4 x i64> {{poison|undef}}, i64 %call, {{i32|i64}} 0
; CHECK: %[[GID_SPLAT:.+]] = shufflevector <4 x i64> %[[GID_SPLATINSERT:.+]], <4 x i64> {{poison|undef}}, <4 x i32> zeroinitializer
; CHECK: %[[GID:.+]] = add <4 x i64> %[[GID_SPLAT]], <i64 0, i64 1, i64 2, i64 3>
; CHECK: %[[CMP3:.+]] = icmp eq <4 x i64> %[[A_SPLAT]], %[[GID]]
; CHECK: %[[NOT_CMP4:.+]] = xor <4 x i1> %[[CMP3]], <i1 true, i1 true, i1 true, i1 true>

; CHECK: %[[IDX:.+]] = sext i32 %a to i64
; CHECK: %[[GEP1:.+]] = getelementptr inbounds i32, ptr %b, i64 %[[IDX]]
; CHECK: call void @__vecz_b_masked_store4_ju3ptrb(i32 11, ptr %[[GEP1]], i1 %{{any_of_mask[0-9]*}})

; CHECK: %[[GEP2:.+]] = getelementptr inbounds i32, ptr %b, i64 42
; CHECK: call void @__vecz_b_masked_store4_ju3ptrb(i32 13, ptr %[[GEP2]], i1 %{{any_of_mask[0-9]*}})

; CHECK: ret void
