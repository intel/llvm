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

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_varying_data_const_value(ptr %in, ptr %out)
; The XOR'd sub-group local IDs
; CHECK: [[XORIDS:%.*]] = xor <4 x i32>
; Which mux sub-group each of the XOR'd sub-group local IDs correspond to
; CHECK-DAG: [[MUXXORIDS:%.*]] = udiv <4 x i32> [[XORIDS]], {{<(i32 4(, )?)+>|splat \(i32 4\)}}
; Which vector group element each of the XOR'd sub-group local IDs correspond to
; CHECK-DAG: [[VECXORIDS:%.*]] = urem <4 x i32> [[XORIDS]], {{<(i32 4(, )?)+>|splat \(i32 4\)}}

; Extract the first XOR'd vector-local sub-group local ID from the vector of vector indices
; CHECK: [[IDXELT0:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 0
; Extract the data element that this XOR'd local ID corresponds to
; CHECK: [[ELT0:%.*]] = extractelement <4 x half> [[DATA:%.*]], i32 [[IDXELT0]]
; Extract the first XOR'd mux-local sub-group local ID from the vector of mux indices
; CHECK: [[ID0:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 0
; Shuffle across any hardware sub-group
; CHECK: [[SHUFF_ELT0:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT0]], i32 [[ID0]])
; Put that result into the final vector
; CHECK: [[SHUFF_VEC0:%.*]] = insertelement <4 x half> poison, half [[SHUFF_ELT0]], i32 0

; And so on for the other shuffle values
; CHECK: [[IDXELT1:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 1
; CHECK: [[ELT1:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT1]]
; CHECK: [[ID1:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 1
; CHECK: [[SHUFF_ELT1:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT1]], i32 [[ID1]])
; CHECK: [[SHUFF_VEC1:%.*]] = insertelement <4 x half> [[SHUFF_VEC0]], half [[SHUFF_ELT1]], i32 1

; CHECK: [[IDXELT2:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 2
; CHECK: [[ELT2:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT2]]
; CHECK: [[ID2:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 2
; CHECK: [[SHUFF_ELT2:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT2]], i32 [[ID2]])
; CHECK: [[SHUFF_VEC2:%.*]] = insertelement <4 x half> [[SHUFF_VEC1]], half [[SHUFF_ELT2]], i32 2

; CHECK: [[IDXELT3:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 3
; CHECK: [[ELT3:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT3]]
; CHECK: [[ID3:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 3
; CHECK: [[SHUFF_ELT3:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT3]], i32 [[ID3]])
; CHECK: [[SHUFF_VEC3:%.*]] = insertelement <4 x half> [[SHUFF_VEC2]], half [[SHUFF_ELT3]], i32 3

; CHECK: store <4 x half> [[SHUFF_VEC3]],
define spir_kernel void @kernel_varying_data_const_value(ptr %in, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds half, ptr %in, i64 %gid
  %data = load half, ptr %arrayidx.in, align 2
  %shuffle1 = call half @__mux_sub_group_shuffle_xor_f16(half %data, i32 4)
  %arrayidx.out = getelementptr inbounds half, ptr %out, i64 %gid
  store half %shuffle1, ptr %arrayidx.out, align 2
  ret void
}

; This should just be the same as the previous kernel. The uniform value doesn't change anything.
; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_varying_data_uniform_value(ptr %in, i32 %val, ptr %out)
; CHECK: [[XORIDS:%.*]] = xor <4 x i32>
; CHECK-DAG: [[MUXXORIDS:%.*]] = udiv <4 x i32> [[XORIDS]], {{<(i32 4(, )?)+>|splat \(i32 4\)}}
; CHECK-DAG: [[VECXORIDS:%.*]] = urem <4 x i32> [[XORIDS]], {{<(i32 4(, )?)+>|splat \(i32 4\)}}
; CHECK: [[IDXELT0:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 0
; CHECK: [[ELT0:%.*]] = extractelement <4 x half> [[DATA:%.*]], i32 [[IDXELT0]]
; CHECK: [[ID0:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 0
; CHECK: [[SHUFF_ELT0:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT0]], i32 [[ID0]])
; CHECK: [[SHUFF_VEC0:%.*]] = insertelement <4 x half> poison, half [[SHUFF_ELT0]], i32 0
; CHECK: [[IDXELT1:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 1
; CHECK: [[ELT1:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT1]]
; CHECK: [[ID1:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 1
; CHECK: [[SHUFF_ELT1:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT1]], i32 [[ID1]])
; CHECK: [[SHUFF_VEC1:%.*]] = insertelement <4 x half> [[SHUFF_VEC0]], half [[SHUFF_ELT1]], i32 1
; CHECK: [[IDXELT2:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 2
; CHECK: [[ELT2:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT2]]
; CHECK: [[ID2:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 2
; CHECK: [[SHUFF_ELT2:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT2]], i32 [[ID2]])
; CHECK: [[SHUFF_VEC2:%.*]] = insertelement <4 x half> [[SHUFF_VEC1]], half [[SHUFF_ELT2]], i32 2
; CHECK: [[IDXELT3:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 3
; CHECK: [[ELT3:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT3]]
; CHECK: [[ID3:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 3
; CHECK: [[SHUFF_ELT3:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT3]], i32 [[ID3]])
; CHECK: [[SHUFF_VEC3:%.*]] = insertelement <4 x half> [[SHUFF_VEC2]], half [[SHUFF_ELT3]], i32 3
; CHECK: store <4 x half> [[SHUFF_VEC3]],
define spir_kernel void @kernel_varying_data_uniform_value(ptr %in, i32 %val, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds half, ptr %in, i64 %gid
  %data = load half, ptr %arrayidx.in, align 2
  %shuffle2 = call half @__mux_sub_group_shuffle_xor_f16(half %data, i32 %val)
  %arrayidx.out = getelementptr inbounds half, ptr %out, i64 %gid
  store half %shuffle2, ptr %arrayidx.out, align 2
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_uniform_data_uniform_value(half %data, i32 %val, ptr %out)
; CHECK: [[SPLATINS:%.*]] = insertelement <4 x half> poison, half %data, i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x half> [[SPLATINS]], <4 x half> poison, <4 x i32> zeroinitializer
; CHECK: store <4 x half> [[SPLAT]]
define spir_kernel void @kernel_uniform_data_uniform_value(half %data, i32 %val, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %shuffle3 = call half @__mux_sub_group_shuffle_xor_f16(half %data, i32 %val)
  %arrayidx.out = getelementptr inbounds half, ptr %out, i64 %gid
  store half %shuffle3, ptr %arrayidx.out, align 2
  ret void
}

; This should just be the same as the previous kernel. The varying value doesn't change anything.
; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_varying_data_varying_value(ptr %in, ptr %vals, ptr %out)
; CHECK: [[XORIDS:%.*]] = xor <4 x i32>
; CHECK-DAG: [[MUXXORIDS:%.*]] = udiv <4 x i32> [[XORIDS]], {{<(i32 4(, )?)+>|splat \(i32 4\)}}
; CHECK-DAG: [[VECXORIDS:%.*]] = urem <4 x i32> [[XORIDS]], {{<(i32 4(, )?)+>|splat \(i32 4\)}}
; CHECK: [[IDXELT0:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 0
; CHECK: [[ELT0:%.*]] = extractelement <4 x half> [[DATA:%.*]], i32 [[IDXELT0]]
; CHECK: [[ID0:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 0
; CHECK: [[SHUFF_ELT0:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT0]], i32 [[ID0]])
; CHECK: [[SHUFF_VEC0:%.*]] = insertelement <4 x half> poison, half [[SHUFF_ELT0]], i32 0
; CHECK: [[IDXELT1:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 1
; CHECK: [[ELT1:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT1]]
; CHECK: [[ID1:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 1
; CHECK: [[SHUFF_ELT1:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT1]], i32 [[ID1]])
; CHECK: [[SHUFF_VEC1:%.*]] = insertelement <4 x half> [[SHUFF_VEC0]], half [[SHUFF_ELT1]], i32 1
; CHECK: [[IDXELT2:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 2
; CHECK: [[ELT2:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT2]]
; CHECK: [[ID2:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 2
; CHECK: [[SHUFF_ELT2:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT2]], i32 [[ID2]])
; CHECK: [[SHUFF_VEC2:%.*]] = insertelement <4 x half> [[SHUFF_VEC1]], half [[SHUFF_ELT2]], i32 2
; CHECK: [[IDXELT3:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 3
; CHECK: [[ELT3:%.*]] = extractelement <4 x half> [[DATA]], i32 [[IDXELT3]]
; CHECK: [[ID3:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 3
; CHECK: [[SHUFF_ELT3:%.*]] = call half @__mux_sub_group_shuffle_f16(half [[ELT3]], i32 [[ID3]])
; CHECK: [[SHUFF_VEC3:%.*]] = insertelement <4 x half> [[SHUFF_VEC2]], half [[SHUFF_ELT3]], i32 3
; CHECK: store <4 x half> [[SHUFF_VEC3]],
define spir_kernel void @kernel_varying_data_varying_value(ptr %in, ptr %vals, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds half, ptr %in, i64 %gid
  %data = load half, ptr %arrayidx.in, align 2
  %arrayidx.vals = getelementptr inbounds i32, ptr %in, i64 %gid
  %val = load i32, ptr %arrayidx.vals, align 4
  %shuffle4 = call half @__mux_sub_group_shuffle_xor_f16(half %data, i32 %val)
  %arrayidx.out = getelementptr inbounds half, ptr %out, i64 %gid
  store half %shuffle4, ptr %arrayidx.out, align 2
  ret void
}

; CHECK-LABEL: define spir_kernel void @__vecz_v4_kernel_varying_vec_data_varying_value(ptr %in, ptr %vals, ptr %out)
; CHECK: [[XORIDS:%.*]] = xor <4 x i32>
; CHECK-DAG: [[MUXXORIDS:%.*]] = udiv <4 x i32> [[XORIDS]], {{<(i32 4(, )?)+>|splat \(i32 4\)}}
; CHECK-DAG: [[VECXORIDS:%.*]] = urem <4 x i32> [[XORIDS]], {{<(i32 4(, )?)+>|splat \(i32 4\)}}

; CHECK: [[IDXELT0:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 0
; CHECK: [[MULIDXELT0:%.*]] = mul i32 [[IDXELT0]], 2
; CHECK: [[MADIDXELT00:%.*]] = add i32 [[MULIDXELT0]], 0
; CHECK: [[ELT00:%.*]] = extractelement <8 x float> [[DATA:%.*]], i32 [[MADIDXELT00]]
; CHECK: [[DATAELT00:%.*]] = insertelement <2 x float> poison, float [[ELT00]], i32 0
; CHECK: [[MADIDXELT01:%.*]] = add i32 [[MULIDXELT0]], 1
; CHECK: [[ELT01:%.*]] = extractelement <8 x float> [[DATA:%.*]], i32 [[MADIDXELT01]]
; CHECK: [[DATAELT01:%.*]] = insertelement <2 x float> [[DATAELT00]], float [[ELT01]], i32 1
; CHECK: [[ID0:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 0
; CHECK: [[SHUFF_ELT0:%.*]] = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> [[DATAELT01]], i32 [[ID0]])
; CHECK: [[SHUFF_RES0:%.*]] = call <8 x float> @llvm.vector.insert.v8f32.v2f32(
; CHECK-SAME:                                      <8 x float> poison, <2 x float> [[SHUFF_ELT0]], i64 0)

; CHECK: [[IDXELT1:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 1
; CHECK: [[MULIDXELT1:%.*]] = mul i32 [[IDXELT1]], 2
; CHECK: [[MADIDXELT10:%.*]] = add i32 [[MULIDXELT1]], 0
; CHECK: [[ELT10:%.*]] = extractelement <8 x float> [[DATA:%.*]], i32 [[MADIDXELT10]]
; CHECK: [[DATAELT10:%.*]] = insertelement <2 x float> poison, float [[ELT10]], i32 0
; CHECK: [[MADIDXELT11:%.*]] = add i32 [[MULIDXELT1]], 1
; CHECK: [[ELT11:%.*]] = extractelement <8 x float> [[DATA:%.*]], i32 [[MADIDXELT11]]
; CHECK: [[DATAELT11:%.*]] = insertelement <2 x float> [[DATAELT10]], float [[ELT11]], i32 1
; CHECK: [[ID1:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 1
; CHECK: [[SHUFF_ELT1:%.*]] = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> [[DATAELT11]], i32 [[ID1]])
; CHECK: [[SHUFF_RES1:%.*]] = call <8 x float> @llvm.vector.insert.v8f32.v2f32(
; CHECK-SAME:                                      <8 x float> [[SHUFF_RES0]], <2 x float> [[SHUFF_ELT1]], i64 2)

; CHECK: [[IDXELT2:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 2
; CHECK: [[MULIDXELT2:%.*]] = mul i32 [[IDXELT2]], 2
; CHECK: [[MADIDXELT20:%.*]] = add i32 [[MULIDXELT2]], 0
; CHECK: [[ELT20:%.*]] = extractelement <8 x float> [[DATA:%.*]], i32 [[MADIDXELT20]]
; CHECK: [[DATAELT20:%.*]] = insertelement <2 x float> poison, float [[ELT20]], i32 0
; CHECK: [[MADIDXELT21:%.*]] = add i32 [[MULIDXELT2]], 1
; CHECK: [[ELT21:%.*]] = extractelement <8 x float> [[DATA:%.*]], i32 [[MADIDXELT21]]
; CHECK: [[DATAELT21:%.*]] = insertelement <2 x float> [[DATAELT20]], float [[ELT21]], i32 1
; CHECK: [[ID2:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 2
; CHECK: [[SHUFF_ELT2:%.*]] = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> [[DATAELT21]], i32 [[ID2]])
; CHECK: [[SHUFF_RES2:%.*]] = call <8 x float> @llvm.vector.insert.v8f32.v2f32(
; CHECK-SAME:                                      <8 x float> [[SHUFF_RES1]], <2 x float> [[SHUFF_ELT2]], i64 4)

; CHECK: [[IDXELT3:%.*]] = extractelement <4 x i32> [[VECXORIDS]], i32 3
; CHECK: [[MULIDXELT3:%.*]] = mul i32 [[IDXELT3]], 2
; CHECK: [[MADIDXELT30:%.*]] = add i32 [[MULIDXELT3]], 0
; CHECK: [[ELT30:%.*]] = extractelement <8 x float> [[DATA:%.*]], i32 [[MADIDXELT30]]
; CHECK: [[DATAELT30:%.*]] = insertelement <2 x float> poison, float [[ELT30]], i32 0
; CHECK: [[MADIDXELT31:%.*]] = add i32 [[MULIDXELT3]], 1
; CHECK: [[ELT31:%.*]] = extractelement <8 x float> [[DATA:%.*]], i32 [[MADIDXELT31]]
; CHECK: [[DATAELT31:%.*]] = insertelement <2 x float> [[DATAELT30]], float [[ELT31]], i32 1
; CHECK: [[ID3:%.*]] = extractelement <4 x i32> [[MUXXORIDS]], i32 3
; CHECK: [[SHUFF_ELT3:%.*]] = call <2 x float> @__mux_sub_group_shuffle_v2f32(<2 x float> [[DATAELT31]], i32 [[ID3]])
; CHECK: [[SHUFF_RES3:%.*]] = call <8 x float> @llvm.vector.insert.v8f32.v2f32(
; CHECK-SAME:                                      <8 x float> [[SHUFF_RES2]], <2 x float> [[SHUFF_ELT3]], i64 6)

; CHECK: store <8 x float> [[SHUFF_RES3]]
define spir_kernel void @kernel_varying_vec_data_varying_value(ptr %in, ptr %vals, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds <2 x float>, ptr %in, i64 %gid
  %data = load <2 x float>, ptr %arrayidx.in, align 8
  %arrayidx.vals = getelementptr inbounds i32, ptr %in, i64 %gid
  %val = load i32, ptr %arrayidx.vals, align 4
  %shuffle5 = call <2 x float> @__mux_sub_group_shuffle_xor_v2f32(<2 x float> %data, i32 %val)
  %arrayidx.out = getelementptr inbounds <2 x float>, ptr %out, i64 %gid
  store <2 x float> %shuffle5, ptr %arrayidx.out, align 8
  ret void
}

declare i64 @__mux_get_global_id(i32)

declare half @__mux_sub_group_shuffle_xor_f16(half, i32)
declare <2 x float> @__mux_sub_group_shuffle_xor_v2f32(<2 x float>, i32)
