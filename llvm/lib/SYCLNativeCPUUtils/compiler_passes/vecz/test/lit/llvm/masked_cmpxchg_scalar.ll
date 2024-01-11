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
  %ret = call { i32, i1 } @__vecz_b_v1_masked_cmpxchg_align4_acquire_monotonic_1_u3ptrjjb(ptr %p, i32 1, i32 2, i1 true)
  ret void
}

declare { i32, i1 } @__vecz_b_v1_masked_cmpxchg_align4_acquire_monotonic_1_u3ptrjjb(ptr %p, i32 %cmp, i32 %newval, i1 %mask)

; CHECK: define { i32, i1 } @__vecz_b_v1_masked_cmpxchg_align4_acquire_monotonic_1_u3ptrjjb(ptr %p, i32 %cmp, i32 %newval, i1 %mask) {
; CHECK: entry:
; CHECK: br label %loopIR

; CHECK: loopIR:
; CHECK: [[RETVAL_PREV:%.*]] = phi i32 [ poison, %entry ], [ [[RETVAL:%.*]], %if.else ]
; CHECK: [[RETSUCC_PREV:%.*]] = phi i1 [ poison, %entry ], [ [[RETSUCC:%.*]], %if.else ]
; CHECK: [[MASKCMP:%.*]] = icmp ne i1 %mask, false
; CHECK: br i1 [[MASKCMP]], label %if.then, label %if.else

; CHECK: if.then:
; CHECK: [[ATOM:%.*]] = cmpxchg ptr %p, i32 %cmp, i32 %newval acquire monotonic, align 4
; CHECK: [[EXT0:%.*]] = extractvalue { i32, i1 } [[ATOM]], 0
; CHECK: [[EXT1:%.*]] = extractvalue { i32, i1 } [[ATOM]], 1
; CHECK: br label %if.else

; CHECK: if.else:
; CHECK: [[RETVAL]] = phi i32 [ [[RETVAL_PREV]], %loopIR ], [ [[EXT0]], %if.then ]
; CHECK: [[RETSUCC]] = phi i1 [ [[RETSUCC_PREV]], %loopIR ], [ [[EXT1]], %if.then ]
; CHECK: [[CMP:%.*]] = icmp ult i32 %{{.*}}, 1
; CHECK: br i1 [[CMP]], label %loopIR, label %exit

; CHECK: exit:
; CHECK: [[INS0:%.*]] = insertvalue { i32, i1 } poison, i32 [[RETVAL]], 0
; CHECK: [[INS1:%.*]] = insertvalue { i32, i1 } [[INS0]], i1 [[RETSUCC]], 1
; CHECK: ret { i32, i1 } [[INS1]]
