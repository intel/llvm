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

; RUN: veczc -vecz-passes=define-builtins,verify -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_fn(ptr %p) {
  %ret = call i32 @__vecz_b_v1_masked_atomicrmw_add_align4_acquire_1_u3ptrjb(ptr %p, i32 1, i1 true)
  ret void
}

declare i32 @__vecz_b_v1_masked_atomicrmw_add_align4_acquire_1_u3ptrjb(ptr %p, i32 %val, i1 %mask)

; CHECK: define i32 @__vecz_b_v1_masked_atomicrmw_add_align4_acquire_1_u3ptrjb(ptr %p, i32 %val, i1 %mask) {
; CHECK: entry:
; CHECK: br label %loopIR

; CHECK: loopIR:
; CHECK: [[RET_PREV:%.*]] = phi i32 [ poison, %entry ], [ [[RET:%.*]], %if.else ]
; CHECK: [[MASKCMP:%.*]] = icmp ne i1 %mask, false
; CHECK: br i1 [[MASKCMP]], label %if.then, label %if.else

; CHECK: if.then:
; CHECK: [[ATOM:%.*]] = atomicrmw add ptr %p, i32 %val acquire, align 4
; CHECK: br label %if.else

; CHECK: if.else:
; CHECK: [[RET]] = phi i32 [ [[RET_PREV]], %loopIR ], [ [[ATOM]], %if.then ]
; CHECK: [[CMP:%.*]] = icmp ult i32 %{{.*}}, 1
; CHECK: br i1 [[CMP]], label %loopIR, label %exit

; CHECK: exit:
; CHECK: ret i32 [[RET]]
