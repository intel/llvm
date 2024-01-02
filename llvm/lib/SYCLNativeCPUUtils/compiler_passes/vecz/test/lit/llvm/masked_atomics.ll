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

; RUN: veczc -w 4 -vecz-passes=cfg-convert,verify,packetizer,define-builtins,verify -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK: define spir_kernel void @__vecz_v4_test_fn(ptr %p)
define spir_kernel void @test_fn(ptr %p) {
entry:
; CHECK: [[SPLAT_PTR_INS:%.*]] = insertelement <4 x ptr> poison, ptr %p, i64 0
; CHECK: [[SPLAT_PTR:%.*]] = shufflevector <4 x ptr> [[SPLAT_PTR_INS]], <4 x ptr> poison, <4 x i32> zeroinitializer
; CHECK: [[CMP:%.*]] = icmp sgt <4 x i64> <i64 3, i64 3, i64 3, i64 3>, 
  %call = call i64 @__mux_get_global_id(i32 0)
  %cmp = icmp sgt i64 3, %call
; CHECK: [[VEC_PTR:%.*]] = getelementptr i32, ptr %p, <4 x i64>
  %wi_p_i32 = getelementptr i32, ptr %p, i64 %call
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
; CHECK: = call <4 x i32> @__vecz_b_v4_masked_atomicrmw_add_align4_acquire_1_Dv4_u3ptrDv4_jDv4_b(
; CHECK-SAME: <4 x ptr> [[SPLAT_PTR]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i1> [[CMP]]
  %old0 = atomicrmw add ptr %p, i32 1 acquire
; CHECK: = call <4 x i32> @__vecz_b_v4_masked_atomicrmw_add_align4_acquire_1_Dv4_u3ptrDv4_jDv4_b(
; CHECK-SAME: <4 x ptr> [[VEC_PTR]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i1> [[CMP]]
  %old1 = atomicrmw add ptr %wi_p_i32, i32 1 acquire
; CHECK: = call <4 x i32> @__vecz_b_v4_masked_atomicrmw_umin_align2_monotonic_1_Dv4_u3ptrDv4_jDv4_b(
; CHECK-SAME: <4 x ptr> [[VEC_PTR]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i1> [[CMP]]
  %old2 = atomicrmw umin ptr %wi_p_i32, i32 1 monotonic, align 2
; CHECK: = call <4 x float> @__vecz_b_v4_masked_atomicrmw_volatile_fmax_align4_seqcst_0_Dv4_u3ptrDv4_fDv4_b(
; CHECK-SAME: <4 x ptr> [[VEC_PTR]], <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x i1> [[CMP]]
  %old3 = atomicrmw volatile fmax ptr %wi_p_i32, float 1.0 syncscope("singlethread") seq_cst
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; CHECK: define <4 x i32> @__vecz_b_v4_masked_atomicrmw_add_align4_acquire_1_Dv4_u3ptrDv4_jDv4_b(<4 x ptr> [[PTRS:%0]], <4 x i32> [[VALS:%1]], <4 x i1> [[MASK:%2]]) [[ATTRS:#[0-9]+]] {
; CHECK: entry:
; CHECK: br label %loopIR

; CHECK: loopIR:
; CHECK: [[IDX:%.*]] = phi i32 [ 0, %entry ], [ [[IDX_NEXT:%.*]], %if.else ]
; CHECK: [[PREV:%.*]] = phi <4 x i32> [ poison, %entry ], [ [[MERGE:%.*]], %if.else ]
; CHECK: [[MASKELT:%.*]] = extractelement <4 x i1> [[MASK]], i32 [[IDX]]
; CHECK: [[MASKCMP:%.*]] = icmp ne i1 [[MASKELT]], false
; CHECK: br i1 [[MASKCMP]], label %if.then, label %if.else

; CHECK: if.then:
; CHECK: [[PTR:%.*]] = extractelement <4 x ptr> [[PTRS]], i32 [[IDX]]
; CHECK: [[VAL:%.*]] = extractelement <4 x i32> [[VALS]], i32 [[IDX]]
; CHECK: [[ATOM:%.*]] = atomicrmw add ptr [[PTR]], i32 [[VAL]] acquire, align 4
; CHECK: [[RET:%.*]] = insertelement <4 x i32> [[PREV]], i32 [[ATOM]], i32 [[IDX]]
; CHECK: br label %if.else

; CHECK: if.else:
; CHECK: [[MERGE]] = phi <4 x i32> [ [[PREV]], %loopIR ], [ [[RET]], %if.then ]
; CHECK: [[IDX_NEXT]] = add i32 [[IDX]], 1

; CHECK: exit:
; CHECK: ret <4 x i32> [[MERGE]]

; Assume that all masked atomicrmw operations follow the logic above. Just
; check that the right atomicrmw instruction is being generated.
; CHECK: define <4 x i32> @__vecz_b_v4_masked_atomicrmw_umin_align2_monotonic_1_Dv4_u3ptrDv4_jDv4_b(<4 x ptr> [[PTRS:%0]], <4 x i32> [[VALS:%1]], <4 x i1> [[MASK:%2]]) [[ATTRS]] {
; CHECK: atomicrmw umin ptr {{%.*}}, i32 {{%.*}} monotonic, align 2


; CHECK: define <4 x float> @__vecz_b_v4_masked_atomicrmw_volatile_fmax_align4_seqcst_0_Dv4_u3ptrDv4_fDv4_b(<4 x ptr> [[PTRS:%0]], <4 x float> [[VALS:%1]], <4 x i1> [[MASK:%2]]) [[ATTRS]] {
; CHECK: atomicrmw volatile fmax ptr {{%.*}}, float {{%.*}} syncscope("singlethread") seq_cst, align 4

; CHECK: attributes [[ATTRS]] = { norecurse nounwind }

declare i64 @__mux_get_global_id(i32)
