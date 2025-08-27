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

; RUN: veczc -w 4 -vecz-passes=packetizer,verify -S \
; RUN:   --pass-remarks-missed=vecz < %s 2>&1 | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

; See @kernel_varying_idx, below
; CHECK: Could not packetize sub-group shuffle %shuffle9

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel(ptr %in, ptr %out)
; CHECK: [[VECIDX:%.*]] = urem i32 %size_minus_1, 4
; CHECK: [[MUXIDX:%.*]] = udiv i32 %size_minus_1, 4
; CHECK: [[VEC:%.*]] = extractelement <4 x i64> {{%.*}}, i32 [[VECIDX]]
; CHECK: [[SHUFFLE:%.*]] = call i64 @__mux_sub_group_shuffle_i64(i64 [[VEC]], i32 [[MUXIDX]])
; CHECK: [[SPLATINS:%.*]] = insertelement <4 x i64> poison, i64 [[SHUFFLE]], i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x i64> [[SPLATINS]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK: store <4 x i64> [[SPLAT]]
define spir_kernel void @kernel(ptr %in, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %size = call i32 @__mux_get_sub_group_size()
  %size_minus_1 = sub i32 %size, 1
  %arrayidx.in = getelementptr inbounds i64, ptr %in, i64 %gid
  %val = load i64, ptr %arrayidx.in, align 8
  %shuffle1 = call i64 @__mux_sub_group_shuffle_i64(i64 %val, i32 %size_minus_1)
  %arrayidx.out = getelementptr inbounds i64, ptr %out, i64 %gid
  store i64 %shuffle1, ptr %arrayidx.out, align 8
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_vec_data(ptr %in, ptr %out)
; CHECK: [[VECIDX:%.*]] = urem i32 %size_minus_1, 4
; CHECK: [[MUXIDX:%.*]] = udiv i32 %size_minus_1, 4
; CHECK: [[BASE:%.*]] = mul i32 %2, 2
; CHECK: [[IDX0:%.*]] = add i32 [[BASE]], 0
; CHECK: [[ELT0:%.*]] = extractelement <8 x float> %1, i32 [[IDX0]]
; CHECK: [[TVEC:%.*]] = insertelement <2 x float> poison, float [[ELT0]], i32 0
; CHECK: [[IDX1:%.*]] = add i32 [[BASE]], 1
; CHECK: [[ELT1:%.*]] = extractelement <8 x float> %1, i32 [[IDX1]]
; CHECK: [[VEC:%.*]] = insertelement <2 x float> [[TVEC]], float [[ELT1]], i32 1
; CHECK: [[SHUFFLE:%.*]] = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> [[VEC]], i32 [[MUXIDX]])
; CHECK: [[SPLAT:%.*]] = shufflevector <2 x float> [[SHUFFLE]], <2 x float> poison,
; CHECK-SAME:                          <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
define spir_kernel void @kernel_vec_data(ptr %in, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %size = call i32 @__mux_get_sub_group_size()
  %size_minus_1 = sub i32 %size, 1
  %arrayidx.in = getelementptr inbounds <2 x float>, ptr %in, i64 %gid
  %val = load <2 x float>, ptr %arrayidx.in, align 8
  %shuffle2 = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> %val, i32 %size_minus_1)
  %arrayidx.out = getelementptr inbounds <2 x float>, ptr %out, i64 %gid
  store <2 x float> %shuffle2, ptr %arrayidx.out, align 8
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_const_idx(ptr %in, ptr %out)
; CHECK: [[VEC:%.*]] = extractelement <4 x i64> {{%.*}}, i32 1
; CHECK: [[SHUFFLE:%.*]] = call i64 @__mux_sub_group_shuffle_i64(i64 [[VEC]], i32 0)
; CHECK: [[SPLATINS:%.*]] = insertelement <4 x i64> poison, i64 [[SHUFFLE]], i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x i64> [[SPLATINS]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK: store <4 x i64> [[SPLAT]]
define spir_kernel void @kernel_const_idx(ptr %in, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds i64, ptr %in, i64 %gid
  %val = load i64, ptr %arrayidx.in, align 8
  %shuffle3 = call i64 @__mux_sub_group_shuffle_i64(i64 %val, i32 1)
  %arrayidx.out = getelementptr inbounds i64, ptr %out, i64 %gid
  store i64 %shuffle3, ptr %arrayidx.out, align 8
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_vec_data_const_idx(ptr %in, ptr %out)
; We're wanting the "1th" sub-group member, which becomes the 2-element vector
; at element index 2
; CHECK: [[VEC:%.*]] = call <2 x float> @llvm.vector.extract.v2f32.v8f32(<8 x float> {{%.*}}, i64 2)
; CHECK: [[SHUFFLE:%.*]] = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> [[VEC]], i32 0)
; CHECK: [[SPLAT:%.*]] = shufflevector <2 x float> [[SHUFFLE]], <2 x float> poison,
; CHECK-SAME:                          <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
; CHECK: store <8 x float> [[SPLAT]]
define spir_kernel void @kernel_vec_data_const_idx(ptr %in, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds <2 x float>, ptr %in, i64 %gid
  %val = load <2 x float>, ptr %arrayidx.in, align 8
  %shuffle4 = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> %val, i32 1)
  %arrayidx.out = getelementptr inbounds <2 x float>, ptr %out, i64 %gid
  store <2 x float> %shuffle4, ptr %arrayidx.out, align 8
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_uniform_data(i64 %val, ptr %out)
; It doesn't matter what sub-group index we choose because the data is uniform.
; Just splat it.
; CHECK: [[SPLATINS:%.*]] = insertelement <4 x i64> poison, i64 %val, i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x i64> [[SPLATINS]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK: store <4 x i64> [[SPLAT]]
define spir_kernel void @kernel_uniform_data(i64 %val, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %size = call i32 @__mux_get_sub_group_size()
  %size_minus_1 = sub i32 %size, 1
  %shuffle5 = call i64 @__mux_sub_group_shuffle_i64(i64 %val, i32 %size_minus_1)
  %arrayidx.out = getelementptr inbounds i64, ptr %out, i64 %gid
  store i64 %shuffle5, ptr %arrayidx.out, align 8
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_uniform_data_varying_idx(i64 %val, ptr %idxs, ptr %out)
; It doesn't matter what sub-group index we choose because the data is uniform.
; Just splat it.
; CHECK: [[SPLATINS:%.*]] = insertelement <4 x i64> poison, i64 %val, i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x i64> [[SPLATINS]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK: store <4 x i64> [[SPLAT]]
define spir_kernel void @kernel_uniform_data_varying_idx(i64 %val, ptr %idxs, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.idxs = getelementptr inbounds i32, ptr %idxs, i64 %gid
  %idx = load i32, ptr %arrayidx.idxs, align 4
  %shuffle6 = call i64 @__mux_sub_group_shuffle_i64(i64 %val, i32 %idx)
  %arrayidx.out = getelementptr inbounds i64, ptr %out, i64 %gid
  store i64 %shuffle6, ptr %arrayidx.out, align 8
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_uniform_vec_data(<2 x float> %val, ptr %out)
; It doesn't matter what sub-group index we choose because the data is uniform.
; Just splat it.
; CHECK: [[SPLAT:%.*]] = shufflevector <2 x float> %val, <2 x float> poison,
; CHECK-SAME:                          <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
; CHECK: store <8 x float> [[SPLAT]]
define spir_kernel void @kernel_uniform_vec_data(<2 x float> %val, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %size = call i32 @__mux_get_sub_group_size()
  %size_minus_1 = sub i32 %size, 1
  %shuffle7 = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> %val, i32 %size_minus_1)
  %arrayidx.out = getelementptr inbounds <2 x float>, ptr %out, i64 %gid
  store <2 x float> %shuffle7, ptr %arrayidx.out, align 8
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_uniform_vec_data_varying_idx(<2 x float> %val, ptr %idxs, ptr %out)
; It doesn't matter what sub-group index we choose because the data is uniform.
; Just splat it.
; CHECK: [[SPLAT:%.*]] = shufflevector <2 x float> %val, <2 x float> poison,
; CHECK-SAME:                          <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
; CHECK: store <8 x float> [[SPLAT]]
define spir_kernel void @kernel_uniform_vec_data_varying_idx(<2 x float> %val, ptr %idxs, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.idxs = getelementptr inbounds i32, ptr %idxs, i64 %gid
  %idx = load i32, ptr %arrayidx.idxs, align 4
  %shuffle8 = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> %val, i32 %idx)
  %arrayidx.out = getelementptr inbounds <2 x float>, ptr %out, i64 %gid
  store <2 x float> %shuffle8, ptr %arrayidx.out, align 8
  ret void
}

; We don't support vectorization of varying indices (for now) - see the check
; above (which is printed before the final IR)
define spir_kernel void @kernel_varying_idx(ptr %in, ptr %idxs, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %size = call i32 @__mux_get_sub_group_size()
  %size_minus_1 = sub i32 %size, 1
  %arrayidx.in = getelementptr inbounds i64, ptr %in, i64 %gid
  %val = load i64, ptr %arrayidx.in, align 8
  %arrayidx.idxs = getelementptr inbounds i32, ptr %idxs, i64 %gid
  %idx = load i32, ptr %arrayidx.idxs, align 4
  %shuffle9 = call i64 @__mux_sub_group_shuffle_i64(i64 %val, i32 %idx)
  %arrayidx.out = getelementptr inbounds i64, ptr %out, i64 %gid
  store i64 %shuffle9, ptr %arrayidx.out, align 8
  ret void
}

declare i64 @__mux_get_global_id(i32)

declare i32 @__mux_get_sub_group_size()

declare i64 @__mux_sub_group_shuffle_i64(i64 %val, i32 %lid)
declare <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> %val, i32 %lid)
