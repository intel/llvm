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

define spir_kernel void @test_fn(<vscale x 1 x ptr> %p) {
  %ret0 = call <vscale x 1 x i32> @__vecz_b_nxv1_vp_masked_atomicrmw_add_align4_acquire_1_u9nxv1u3ptru5nxv1ju5nxv1b(<vscale x 1 x ptr> %p, <vscale x 1 x i32> zeroinitializer, <vscale x 1 x i1> zeroinitializer, i32 4)
  %ret1 = call { <vscale x 1 x i32>, <vscale x 1 x i1> } @__vecz_b_nxv1_vp_masked_cmpxchg_align4_acquire_acquire_1_u9nxv1u3ptru5nxv1ju5nxv1ju5nxv1b(<vscale x 1 x ptr> %p, <vscale x 1 x i32> zeroinitializer, <vscale x 1 x i32> zeroinitializer, <vscale x 1 x i1> zeroinitializer, i32 4)
  ret void
}

declare <vscale x 1 x i32> @__vecz_b_nxv1_vp_masked_atomicrmw_add_align4_acquire_1_u9nxv1u3ptru5nxv1ju5nxv1b(<vscale x 1 x ptr> %p, <vscale x 1 x i32> %val, <vscale x 1 x i1> %mask, i32 %vl)

declare { <vscale x 1 x i32>, <vscale x 1 x i1> } @__vecz_b_nxv1_vp_masked_cmpxchg_align4_acquire_acquire_1_u9nxv1u3ptru5nxv1ju5nxv1ju5nxv1b(<vscale x 1 x ptr> %p, <vscale x 1 x i32> %cmp, <vscale x 1 x i32> %newval, <vscale x 1 x i1> %mask, i32 %vl)

; CHECK: define <vscale x 1 x i32> @__vecz_b_nxv1_vp_masked_atomicrmw_add_align4_acquire_1_u9nxv1u3ptru5nxv1ju5nxv1b(<vscale x 1 x ptr> %p, <vscale x 1 x i32> %val, <vscale x 1 x i1> %mask, i32 %vl) {
; CHECK: entry:
; CHECK: [[VLZERO:%.*]] = icmp eq i32 %vl, 0
; CHECK: br i1 [[VLZERO]], label %earlyexit, label %loopentry

; CHECK: earlyexit:
; CHECK: ret <vscale x 1 x i32> poison

; CHECK: loopentry:
; CHECK: br label %loopIR

; CHECK: loopIR:
; CHECK: [[IDX:%.*]] = phi i32 [ 0, %loopentry ], [ [[INC:%.*]], %if.else ]
; CHECK: [[RET_PREV:%.*]] = phi <vscale x 1 x i32> [ poison, %loopentry ], [ [[MERGE:%.*]], %if.else ]
; CHECK: [[MASKELT:%.*]] = extractelement <vscale x 1 x i1> %mask, i32 [[IDX]]
; CHECK: [[MASKCMP:%.*]] = icmp ne i1 [[MASKELT]], false
; CHECK: br i1 [[MASKCMP]], label %if.then, label %if.else

; CHECK: if.then:
; CHECK: [[PTR:%.*]] = extractelement <vscale x 1 x ptr> %p, i32 [[IDX]]
; CHECK: [[VAL:%.*]] = extractelement <vscale x 1 x i32> %val, i32 [[IDX]]
; CHECK: [[ATOM:%.*]] = atomicrmw add ptr [[PTR]], i32 [[VAL]] acquire, align 4
; CHECK: [[RET_NEXT:%.*]] = insertelement <vscale x 1 x i32> [[RET_PREV]], i32 [[ATOM]], i32 [[IDX]]
; CHECK: br label %if.else

; CHECK: if.else:
; CHECK: [[MERGE:%.*]] = phi <vscale x 1 x i32> [ [[RET_PREV]], %loopIR ], [ [[RET_NEXT]], %if.then ]
; CHECK: [[INC]] = add i32 [[IDX]], 1
; CHECK: [[CMP:%.*]] = icmp ult i32 [[INC]], %vl
; CHECK: br i1 [[CMP]], label %loopIR, label %exit

; CHECK: exit:
; CHECK: ret <vscale x 1 x i32> [[MERGE]]

; CHECK: define { <vscale x 1 x i32>, <vscale x 1 x i1> } @__vecz_b_nxv1_vp_masked_cmpxchg_align4_acquire_acquire_1_u9nxv1u3ptru5nxv1ju5nxv1ju5nxv1b(<vscale x 1 x ptr> %p, <vscale x 1 x i32> %cmp, <vscale x 1 x i32> %newval, <vscale x 1 x i1> %mask, i32 %vl) {
; CHECK: entry:
; CHECK: [[VLZERO:%.*]] = icmp eq i32 %vl, 0
; CHECK: br i1 [[VLZERO]], label %earlyexit, label %loopentry

; CHECK: earlyexit:
; CHECK: ret { <vscale x 1 x i32>, <vscale x 1 x i1> } poison

; CHECK: loopentry:
; CHECK: br label %loopIR

; CHECK: loopIR:
; CHECK: [[IDX:%.*]] = phi i32 [ 0, %loopentry ], [ [[INC:%.*]], %if.else ]
; CHECK: [[RET_PREV:%.*]] = phi <vscale x 1 x i32> [ poison, %loopentry ], [ [[MERGE:%.*]], %if.else ]
; CHECK: [[SUCCESS_PREV:%.*]] = phi <vscale x 1 x i1> [ poison, %loopentry ], [ [[MERGE_SUCCESS:%.*]], %if.else ]
; CHECK: [[MASKELT:%.*]] = extractelement <vscale x 1 x i1> %mask, i32 [[IDX]]
; CHECK: [[MASKCMP:%.*]] = icmp ne i1 [[MASKELT]], false
; CHECK: br i1 [[MASKCMP]], label %if.then, label %if.else

; CHECK: if.then:
; CHECK: [[PTR:%.*]] = extractelement <vscale x 1 x ptr> %p, i32 [[IDX]]
; CHECK: [[CMP:%.*]] = extractelement <vscale x 1 x i32> %cmp, i32 [[IDX]]
; CHECK: [[NEWVAL:%.*]] = extractelement <vscale x 1 x i32> %newval, i32 [[IDX]]
; CHECK: [[ATOM:%.*]] = cmpxchg ptr [[PTR]], i32 [[CMP]], i32 [[NEWVAL]] acquire acquire, align 4
; CHECK: [[EXT0:%.*]] = extractvalue { i32, i1 } [[ATOM]], 0
; CHECK: [[RET:%.*]] = insertelement <vscale x 1 x i32> [[RET_PREV]], i32 [[EXT0]], i32 [[IDX]]
; CHECK: [[EXT1:%.*]] = extractvalue { i32, i1 } [[ATOM]], 1
; CHECK: [[SUCCESS:%.*]] = insertelement <vscale x 1 x i1> [[SUCCESS_PREV]], i1 [[EXT1]], i32 [[IDX]]
; CHECK: br label %if.else

; CHECK: if.else:
; CHECK: [[MERGE:%.*]] = phi <vscale x 1 x i32> [ [[RET_PREV]], %loopIR ], [ [[RET]], %if.then ]
; CHECK: [[MERGE_SUCCESS:%.*]] = phi <vscale x 1 x i1> [ [[SUCCESS_PREV]], %loopIR ], [ [[SUCCESS]], %if.then ]
; CHECK: [[INC]] = add i32 [[IDX]], 1
; CHECK: [[CMP:%.*]] = icmp ult i32 [[INC]], %vl
; CHECK: br i1 [[CMP]], label %loopIR, label %exit

; CHECK: exit:
; CHECK: [[RETTMP:%.*]] = insertvalue { <vscale x 1 x i32>, <vscale x 1 x i1> } poison, <vscale x 1 x i32> [[MERGE]], 0
; CHECK: [[RETVAL:%.*]] = insertvalue { <vscale x 1 x i32>, <vscale x 1 x i1> } [[RETTMP]], <vscale x 1 x i1> [[MERGE_SUCCESS]], 1
; CHECK: ret { <vscale x 1 x i32>, <vscale x 1 x i1> } [[RETVAL]]
