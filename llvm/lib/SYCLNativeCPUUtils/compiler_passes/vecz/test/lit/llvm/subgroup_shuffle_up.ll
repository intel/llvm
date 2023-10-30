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

; RUN: veczc -w 4 -vecz-passes=packetizer,verify -S \
; RUN:   --pass-remarks-missed=vecz < %s 2>&1 | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel(ptr %lhsptr, ptr %rhsptr, ptr %out)
; CHECK: [[LHS:%.*]] = load <4 x float>, ptr %arrayidx.lhs, align 4
; CHECK: [[RHS:%.*]] = load <4 x float>, ptr %arrayidx.rhs, align 4

; CHECK: [[DELTAS:%.*]] = sub <4 x i32> {{%.*}}, <i32 1, i32 1, i32 1, i32 1>
; CHECK: [[QUOTIENT:%.*]] = sdiv <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>
; CHECK: [[REMAINDER:%.*]] = srem <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>

; CHECK: [[ARGXOR:%.*]] = xor <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>
; CHECK: [[SIGNDIFF:%.*]] = icmp slt <4 x i32> [[ARGXOR]], zeroinitializer
; CHECK: [[REMNONZERO:%.*]] = icmp ne <4 x i32> [[REMAINDER]], zeroinitializer
; CHECK: [[CONDITION:%.*]] = and <4 x i1> [[REMNONZERO]], [[SIGNDIFF]]

; CHECK: [[MIN1:%.*]] = sub <4 x i32> [[QUOTIENT]], <i32 1, i32 1, i32 1, i32 1>
; CHECK: [[PLUSR:%.*]] = add <4 x i32> [[REMAINDER]], <i32 4, i32 4, i32 4, i32 4> 

; CHECK: [[MUXIDS:%.*]] = select <4 x i1> [[CONDITION]], <4 x i32> [[MIN1]], <4 x i32> [[QUOTIENT]]
; CHECK: [[VECELTS:%.*]] = select <4 x i1> [[CONDITION]], <4 x i32> [[PLUSR]], <4 x i32> [[REMAINDER]]

; CHECK: [[MUXDELTAS:%.*]] = sub <4 x i32> {{%.*}}, [[MUXIDS]]

; CHECK: [[DELTA0:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 0
; CHECK: [[SHUFF0:%.*]] = call <4 x float> @__mux_sub_group_shuffle_up_v4f32(
; CHECK-SAME:                      <4 x float> [[LHS]], <4 x float> [[RHS]], i32 [[DELTA0]])
; CHECK: [[VECIDX0:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 0
; CHECK: [[ELT0:%.*]] = extractelement <4 x float> [[SHUFF0]], i32 [[VECIDX0]]

; CHECK: [[DELTA1:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 1
; CHECK: [[SHUFF1:%.*]] = call <4 x float> @__mux_sub_group_shuffle_up_v4f32(
; CHECK-SAME:                      <4 x float> [[LHS]], <4 x float> [[RHS]], i32 [[DELTA1]])
; CHECK: [[VECIDX1:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 1
; CHECK: [[ELT1:%.*]] = extractelement <4 x float> [[SHUFF1]], i32 [[VECIDX1]]

; CHECK: [[DELTA2:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 2
; CHECK: [[SHUFF2:%.*]] = call <4 x float> @__mux_sub_group_shuffle_up_v4f32(
; CHECK-SAME:                      <4 x float> [[LHS]], <4 x float> [[RHS]], i32 [[DELTA2]])
; CHECK: [[VECIDX2:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 2
; CHECK: [[ELT2:%.*]] = extractelement <4 x float> [[SHUFF2]], i32 [[VECIDX2]]

; CHECK: [[DELTA3:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 3
; CHECK: [[SHUFF3:%.*]] = call <4 x float> @__mux_sub_group_shuffle_up_v4f32(
; CHECK-SAME:                      <4 x float> [[LHS]], <4 x float> [[RHS]], i32 [[DELTA3]])
; CHECK: [[VECIDX3:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 3
; CHECK: [[ELT3:%.*]] = extractelement <4 x float> [[SHUFF3]], i32 [[VECIDX3]]
define spir_kernel void @kernel(ptr %lhsptr, ptr %rhsptr, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.lhs = getelementptr inbounds float, ptr %lhsptr, i64 %gid
  %lhs = load float, ptr %arrayidx.lhs, align 4
  %arrayidx.rhs = getelementptr inbounds float, ptr %rhsptr, i64 %gid
  %rhs = load float, ptr %arrayidx.rhs, align 4
  %shuffle_up = call float @__mux_sub_group_shuffle_up_f32(float %lhs, float %rhs, i32 1)
  %arrayidx.out = getelementptr inbounds float, ptr %out, i64 %gid
  store float %shuffle_up, ptr %arrayidx.out, align 8
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_vec_data(ptr %lhsptr, ptr %rhsptr, ptr %out)
; CHECK: [[DELTAS:%.*]] = sub <4 x i32> {{%.*}}, <i32 2, i32 2, i32 2, i32 2>
; CHECK: [[QUOTIENT:%.*]] = sdiv <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>
; CHECK: [[REMAINDER:%.*]] = srem <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>

; CHECK: [[ARGXOR:%.*]] = xor <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>
; CHECK: [[SIGNDIFF:%.*]] = icmp slt <4 x i32> [[ARGXOR]], zeroinitializer
; CHECK: [[REMNONZERO:%.*]] = icmp ne <4 x i32> [[REMAINDER]], zeroinitializer
; CHECK: [[CONDITION:%.*]] = and <4 x i1> [[REMNONZERO]], [[SIGNDIFF]]

; CHECK: [[MIN1:%.*]] = sub <4 x i32> [[QUOTIENT]], <i32 1, i32 1, i32 1, i32 1>
; CHECK: [[PLUSR:%.*]] = add <4 x i32> [[REMAINDER]], <i32 4, i32 4, i32 4, i32 4> 

; CHECK: [[MUXIDS:%.*]] = select <4 x i1> [[CONDITION]], <4 x i32> [[MIN1]], <4 x i32> [[QUOTIENT]]
; CHECK: [[VECELTS:%.*]] = select <4 x i1> [[CONDITION]], <4 x i32> [[PLUSR]], <4 x i32> [[REMAINDER]]

; CHECK: [[MUXDELTAS:%.*]] = sub <4 x i32> {{%.*}}, [[MUXIDS]]

; CHECK: [[DELTA0:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 0
; CHECK: [[SHUFF0:%.*]] = call <16 x i8> @__mux_sub_group_shuffle_up_v16i8(
; CHECK-SAME:                      <16 x i8> [[LHS:%.*]], <16 x i8> [[RHS:%.*]], i32 [[DELTA0]])
; CHECK: [[SUBVECIDX0:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 0
; CHECK: [[ELTBASE0:%.*]] = mul i32 [[SUBVECIDX0]], 4
; CHECK: [[VECIDX00:%.*]] = add i32 [[ELTBASE0]], 0
; CHECK: [[ELT00:%.*]] = extractelement <16 x i8> [[SHUFF0]], i32 [[VECIDX00]]
; CHECK: [[VEC00:%.*]] = insertelement <4 x i8> undef, i8 [[ELT00]], i32 0
; CHECK: [[VECIDX01:%.*]] = add i32 [[ELTBASE0]], 1
; CHECK: [[ELT01:%.*]] = extractelement <16 x i8> [[SHUFF0]], i32 [[VECIDX01]]
; CHECK: [[VEC01:%.*]] = insertelement <4 x i8> [[VEC00]], i8 [[ELT01]], i32 1
; CHECK: [[VECIDX02:%.*]] = add i32 [[ELTBASE0]], 2
; CHECK: [[ELT02:%.*]] = extractelement <16 x i8> [[SHUFF0]], i32 [[VECIDX02]]
; CHECK: [[VEC02:%.*]] = insertelement <4 x i8> [[VEC01]], i8 [[ELT02]], i32 2
; CHECK: [[VECIDX03:%.*]] = add i32 [[ELTBASE0]], 3
; CHECK: [[ELT03:%.*]] = extractelement <16 x i8> [[SHUFF0]], i32 [[VECIDX03]]
; CHECK: [[VEC03:%.*]] = insertelement <4 x i8> [[VEC02]], i8 [[ELT03]], i32 3

; CHECK: [[DELTA1:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 1
; CHECK: [[SHUFF1:%.*]] = call <16 x i8> @__mux_sub_group_shuffle_up_v16i8(
; CHECK-SAME:                      <16 x i8> [[LHS]], <16 x i8> [[RHS]], i32 [[DELTA1]])
; CHECK: [[SUBVECIDX1:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 1
; CHECK: [[ELTBASE1:%.*]] = mul i32 [[SUBVECIDX1]], 4
; CHECK: [[VECIDX10:%.*]] = add i32 [[ELTBASE1]], 0
; CHECK: [[ELT10:%.*]] = extractelement <16 x i8> [[SHUFF1]], i32 [[VECIDX10]]
; CHECK: [[VEC10:%.*]] = insertelement <4 x i8> undef, i8 [[ELT10]], i32 0
; CHECK: [[VECIDX11:%.*]] = add i32 [[ELTBASE1]], 1
; CHECK: [[ELT11:%.*]] = extractelement <16 x i8> [[SHUFF1]], i32 [[VECIDX11]]
; CHECK: [[VEC11:%.*]] = insertelement <4 x i8> [[VEC10]], i8 [[ELT11]], i32 1
; CHECK: [[VECIDX12:%.*]] = add i32 [[ELTBASE1]], 2
; CHECK: [[ELT12:%.*]] = extractelement <16 x i8> [[SHUFF1]], i32 [[VECIDX12]]
; CHECK: [[VEC12:%.*]] = insertelement <4 x i8> [[VEC11]], i8 [[ELT12]], i32 2
; CHECK: [[VECIDX13:%.*]] = add i32 [[ELTBASE1]], 3
; CHECK: [[ELT13:%.*]] = extractelement <16 x i8> [[SHUFF1]], i32 [[VECIDX13]]
; CHECK: [[VEC13:%.*]] = insertelement <4 x i8> [[VEC12]], i8 [[ELT13]], i32 3

; CHECK: [[DELTA2:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 2
; CHECK: [[SHUFF2:%.*]] = call <16 x i8> @__mux_sub_group_shuffle_up_v16i8(
; CHECK-SAME:                      <16 x i8> [[LHS]], <16 x i8> [[RHS]], i32 [[DELTA2]])
; CHECK: [[SUBVECIDX2:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 2
; CHECK: [[ELTBASE2:%.*]] = mul i32 [[SUBVECIDX2]], 4
; CHECK: [[VECIDX20:%.*]] = add i32 [[ELTBASE2]], 0
; CHECK: [[ELT20:%.*]] = extractelement <16 x i8> [[SHUFF2]], i32 [[VECIDX20]]
; CHECK: [[VEC20:%.*]] = insertelement <4 x i8> undef, i8 [[ELT20]], i32 0
; CHECK: [[VECIDX21:%.*]] = add i32 [[ELTBASE2]], 1
; CHECK: [[ELT21:%.*]] = extractelement <16 x i8> [[SHUFF2]], i32 [[VECIDX21]]
; CHECK: [[VEC21:%.*]] = insertelement <4 x i8> [[VEC20]], i8 [[ELT21]], i32 1
; CHECK: [[VECIDX22:%.*]] = add i32 [[ELTBASE2]], 2
; CHECK: [[ELT22:%.*]] = extractelement <16 x i8> [[SHUFF2]], i32 [[VECIDX22]]
; CHECK: [[VEC22:%.*]] = insertelement <4 x i8> [[VEC21]], i8 [[ELT22]], i32 2
; CHECK: [[VECIDX23:%.*]] = add i32 [[ELTBASE2]], 3
; CHECK: [[ELT23:%.*]] = extractelement <16 x i8> [[SHUFF2]], i32 [[VECIDX23]]
; CHECK: [[VEC23:%.*]] = insertelement <4 x i8> [[VEC22]], i8 [[ELT23]], i32 3

; CHECK: [[DELTA3:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 3
; CHECK: [[SHUFF3:%.*]] = call <16 x i8> @__mux_sub_group_shuffle_up_v16i8(
; CHECK-SAME:                      <16 x i8> [[LHS]], <16 x i8> [[RHS]], i32 [[DELTA3]])
; CHECK: [[SUBVECIDX3:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 3
; CHECK: [[ELTBASE3:%.*]] = mul i32 [[SUBVECIDX3]], 4
; CHECK: [[VECIDX30:%.*]] = add i32 [[ELTBASE3]], 0
; CHECK: [[ELT30:%.*]] = extractelement <16 x i8> [[SHUFF3]], i32 [[VECIDX30]]
; CHECK: [[VEC30:%.*]] = insertelement <4 x i8> undef, i8 [[ELT30]], i32 0
; CHECK: [[VECIDX31:%.*]] = add i32 [[ELTBASE3]], 1
; CHECK: [[ELT31:%.*]] = extractelement <16 x i8> [[SHUFF3]], i32 [[VECIDX31]]
; CHECK: [[VEC31:%.*]] = insertelement <4 x i8> [[VEC30]], i8 [[ELT31]], i32 1
; CHECK: [[VECIDX32:%.*]] = add i32 [[ELTBASE3]], 2
; CHECK: [[ELT32:%.*]] = extractelement <16 x i8> [[SHUFF3]], i32 [[VECIDX32]]
; CHECK: [[VEC32:%.*]] = insertelement <4 x i8> [[VEC31]], i8 [[ELT32]], i32 2
; CHECK: [[VECIDX33:%.*]] = add i32 [[ELTBASE3]], 3
; CHECK: [[ELT33:%.*]] = extractelement <16 x i8> [[SHUFF3]], i32 [[VECIDX33]]
; CHECK: [[VEC33:%.*]] = insertelement <4 x i8> [[VEC32]], i8 [[ELT33]], i32 3
define spir_kernel void @kernel_vec_data(ptr %lhsptr, ptr %rhsptr, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.lhs = getelementptr inbounds <4 x i8>, ptr %lhsptr, i64 %gid
  %lhs = load <4 x i8>, ptr %arrayidx.lhs, align 4
  %arrayidx.rhs = getelementptr inbounds <4 x i8>, ptr %rhsptr, i64 %gid
  %rhs = load <4 x i8>, ptr %arrayidx.rhs, align 4
  %shuffle_up = call <4 x i8> @__mux_sub_group_shuffle_up_v4i8(<4 x i8> %lhs, <4 x i8> %rhs, i32 2)
  %arrayidx.out = getelementptr inbounds <4 x i8>, ptr %out, i64 %gid
  store <4 x i8> %shuffle_up, ptr %arrayidx.out, align 4
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_varying_delta(ptr %lhsptr, ptr %rhsptr, ptr %deltaptr, ptr %out)
; CHECK: [[LHS:%.*]] = load <4 x float>, ptr %arrayidx.lhs, align 4
; CHECK: [[RHS:%.*]] = load <4 x float>, ptr %arrayidx.rhs, align 4
; CHECK: [[DELTALD:%.*]] = load <4 x i32>, ptr %arrayidx.deltas, align 4

; CHECK: [[DELTAS:%.*]] = sub <4 x i32> {{%.*}}, [[DELTALD]]
; CHECK: [[QUOTIENT:%.*]] = sdiv <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>
; CHECK: [[REMAINDER:%.*]] = srem <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>

; CHECK: [[ARGXOR:%.*]] = xor <4 x i32> [[DELTAS]], <i32 4, i32 4, i32 4, i32 4>
; CHECK: [[SIGNDIFF:%.*]] = icmp slt <4 x i32> [[ARGXOR]], zeroinitializer
; CHECK: [[REMNONZERO:%.*]] = icmp ne <4 x i32> [[REMAINDER]], zeroinitializer
; CHECK: [[CONDITION:%.*]] = and <4 x i1> [[REMNONZERO]], [[SIGNDIFF]]

; CHECK: [[MIN1:%.*]] = sub <4 x i32> [[QUOTIENT]], <i32 1, i32 1, i32 1, i32 1>
; CHECK: [[PLUSR:%.*]] = add <4 x i32> [[REMAINDER]], <i32 4, i32 4, i32 4, i32 4> 

; CHECK: [[MUXIDS:%.*]] = select <4 x i1> [[CONDITION]], <4 x i32> [[MIN1]], <4 x i32> [[QUOTIENT]]
; CHECK: [[VECELTS:%.*]] = select <4 x i1> [[CONDITION]], <4 x i32> [[PLUSR]], <4 x i32> [[REMAINDER]]

; CHECK: [[MUXDELTAS:%.*]] = sub <4 x i32> {{%.*}}, [[MUXIDS]]

; CHECK: [[DELTA0:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 0
; CHECK: [[SHUFF0:%.*]] = call <4 x float> @__mux_sub_group_shuffle_up_v4f32(
; CHECK-SAME:                      <4 x float> [[LHS]], <4 x float> [[RHS]], i32 [[DELTA0]])
; CHECK: [[VECIDX0:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 0
; CHECK: [[ELT0:%.*]] = extractelement <4 x float> [[SHUFF0]], i32 [[VECIDX0]]

; CHECK: [[DELTA1:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 1
; CHECK: [[SHUFF1:%.*]] = call <4 x float> @__mux_sub_group_shuffle_up_v4f32(
; CHECK-SAME:                      <4 x float> [[LHS]], <4 x float> [[RHS]], i32 [[DELTA1]])
; CHECK: [[VECIDX1:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 1
; CHECK: [[ELT1:%.*]] = extractelement <4 x float> [[SHUFF1]], i32 [[VECIDX1]]

; CHECK: [[DELTA2:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 2
; CHECK: [[SHUFF2:%.*]] = call <4 x float> @__mux_sub_group_shuffle_up_v4f32(
; CHECK-SAME:                      <4 x float> [[LHS]], <4 x float> [[RHS]], i32 [[DELTA2]])
; CHECK: [[VECIDX2:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 2
; CHECK: [[ELT2:%.*]] = extractelement <4 x float> [[SHUFF2]], i32 [[VECIDX2]]

; CHECK: [[DELTA3:%.*]] = extractelement <4 x i32> [[MUXDELTAS]], i32 3
; CHECK: [[SHUFF3:%.*]] = call <4 x float> @__mux_sub_group_shuffle_up_v4f32(
; CHECK-SAME:                      <4 x float> [[LHS]], <4 x float> [[RHS]], i32 [[DELTA3]])
; CHECK: [[VECIDX3:%.*]] = extractelement <4 x i32> [[VECELTS]], i32 3
; CHECK: [[ELT3:%.*]] = extractelement <4 x float> [[SHUFF3]], i32 [[VECIDX3]]
define spir_kernel void @kernel_varying_delta(ptr %lhsptr, ptr %rhsptr, ptr %deltaptr, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.lhs = getelementptr inbounds float, ptr %lhsptr, i64 %gid
  %lhs = load float, ptr %arrayidx.lhs, align 4
  %arrayidx.rhs = getelementptr inbounds float, ptr %rhsptr, i64 %gid
  %rhs = load float, ptr %arrayidx.rhs, align 4
  %arrayidx.deltas = getelementptr inbounds i32, ptr %deltaptr, i64 %gid
  %delta = load i32, ptr %arrayidx.deltas, align 4
  %shuffle_up = call float @__mux_sub_group_shuffle_up_f32(float %lhs, float %rhs, i32 %delta)
  %arrayidx.out = getelementptr inbounds float, ptr %out, i64 %gid
  store float %shuffle_up, ptr %arrayidx.out, align 8
  ret void
}

declare i64 @__mux_get_global_id(i32)

declare float @__mux_sub_group_shuffle_up_f32(float %prev, float %curr, i32 %delta)
declare <4 x i8> @__mux_sub_group_shuffle_up_v4i8(<4 x i8> %prev, <4 x i8> %curr, i32 %delta)
