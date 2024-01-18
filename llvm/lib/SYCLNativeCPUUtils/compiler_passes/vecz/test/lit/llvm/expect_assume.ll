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

; RUN: veczc -vecz-simd-width=4 -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @__mux_get_global_id(i32)

declare void @llvm.assume(i1)
declare i32 @llvm.expect.i32(i32, i32)

; CHECK: define spir_kernel void @__vecz_v4_assume(
; CHECK: [[A:%.*]] = load <4 x i32>, ptr %arrayidxa, align 4
; CHECK: [[B:%.*]] = load <4 x i32>, ptr %arrayidxb, align 4
; CHECK: [[SUM:%.*]] = add <4 x i32> %0, %1
; CHECK: [[CMP:%.*]] = icmp sgt <4 x i32> [[SUM]], zeroinitializer
; CHECK: [[E0:%.*]] = extractelement <4 x i1> [[CMP]], i64 0
; CHECK: [[E1:%.*]] = extractelement <4 x i1> [[CMP]], i64 1
; CHECK: [[E2:%.*]] = extractelement <4 x i1> [[CMP]], i64 2
; CHECK: [[E3:%.*]] = extractelement <4 x i1> [[CMP]], i64 3
; CHECK: call void @llvm.assume(i1 [[E0]])
; CHECK: call void @llvm.assume(i1 [[E1]])
; CHECK: call void @llvm.assume(i1 [[E2]])
; CHECK: call void @llvm.assume(i1 [[E3]])
; CHECK: store <4 x i32> [[SUM]], ptr %arrayidxz, align 4
define spir_kernel void @assume(ptr %aptr, ptr %bptr, ptr %zptr) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxa = getelementptr inbounds i32, ptr %aptr, i64 %idx
  %arrayidxb = getelementptr inbounds i32, ptr %bptr, i64 %idx
  %arrayidxz = getelementptr inbounds i32, ptr %zptr, i64 %idx
  %a = load i32, ptr %arrayidxa, align 4
  %b = load i32, ptr %arrayidxb, align 4
  %sum = add i32 %a, %b
  %cond = icmp sgt i32 %sum, 0
  call void @llvm.assume(i1 %cond)
  store i32 %sum, ptr %arrayidxz, align 4
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_expect(
; CHECK: [[A:%.*]] = load <4 x i32>, ptr %arrayidxa, align 4
; CHECK: [[B:%.*]] = load <4 x i32>, ptr %arrayidxb, align 4
; CHECK: [[SUM:%.*]] = add <4 x i32> %0, %1
; CHECK: [[E0:%.*]] = extractelement <4 x i32> [[SUM]], i64 0
; CHECK: [[E1:%.*]] = extractelement <4 x i32> [[SUM]], i64 1
; CHECK: [[E2:%.*]] = extractelement <4 x i32> [[SUM]], i64 2
; CHECK: [[E3:%.*]] = extractelement <4 x i32> [[SUM]], i64 3
; CHECK: [[EX0:%.*]] = call i32 @llvm.expect.i32(i32 [[E0]], i32 42)
; CHECK: [[EX1:%.*]] = call i32 @llvm.expect.i32(i32 [[E1]], i32 42)
; CHECK: [[EX2:%.*]] = call i32 @llvm.expect.i32(i32 [[E2]], i32 42)
; CHECK: [[EX3:%.*]] = call i32 @llvm.expect.i32(i32 [[E3]], i32 42)
; CHECK: [[C0:%.*]] = insertelement <4 x i32> {{undef|poison}}, i32 [[EX0]], i64 0
; CHECK: [[C1:%.*]]  = insertelement <4 x i32> [[C0]], i32 [[EX1]], i64 1
; CHECK: [[C2:%.*]]  = insertelement <4 x i32> [[C1]], i32 [[EX2]], i64 2
; CHECK: [[C3:%.*]]  = insertelement <4 x i32> [[C2]], i32 [[EX3]], i64 3
; CHECK: store <4 x i32> [[C3]], ptr %arrayidxz, align 4

define spir_kernel void @expect(ptr %aptr, ptr %bptr, ptr %zptr) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidxa = getelementptr inbounds i32, ptr %aptr, i64 %idx
  %arrayidxb = getelementptr inbounds i32, ptr %bptr, i64 %idx
  %arrayidxz = getelementptr inbounds i32, ptr %zptr, i64 %idx
  %a = load i32, ptr %arrayidxa, align 4
  %b = load i32, ptr %arrayidxb, align 4
  %sum = add i32 %a, %b
  %cond = icmp sgt i32 %sum, 0
  %v = call i32 @llvm.expect.i32(i32 %sum, i32 42)
  store i32 %v, ptr %arrayidxz, align 4
  ret void
}
