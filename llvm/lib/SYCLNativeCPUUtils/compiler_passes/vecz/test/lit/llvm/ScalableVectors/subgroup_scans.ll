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

; RUN: veczc -vecz-scalable -vecz-simd-width=4 -S -vecz-passes=packetizer < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i64 @__mux_get_global_id(i32)

declare spir_func i32 @__mux_sub_group_scan_inclusive_add_i32(i32)
declare spir_func i64 @__mux_sub_group_scan_inclusive_add_i64(i64)
declare spir_func float @__mux_sub_group_scan_inclusive_fadd_f32(float)

declare spir_func i32 @__mux_sub_group_scan_inclusive_smin_i32(i32)
declare spir_func i32 @__mux_sub_group_scan_inclusive_umin_i32(i32)
declare spir_func i32 @__mux_sub_group_scan_inclusive_smax_i32(i32)
declare spir_func i32 @__mux_sub_group_scan_inclusive_umax_i32(i32)
declare spir_func float @__mux_sub_group_scan_inclusive_fmin_f32(float)
declare spir_func float @__mux_sub_group_scan_inclusive_fmax_f32(float)

define spir_kernel void @reduce_scan_incl_add_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @__mux_sub_group_scan_inclusive_add_i32(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_add_i32(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4j(<vscale x 4 x i32> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call i32 @__mux_sub_group_scan_exclusive_add_i32(i32 [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[HEAD]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = add <vscale x 4 x i32> [[SCAN]], [[SPLAT]]
; CHECK: store <vscale x 4 x i32> [[FINAL]],
}

define spir_kernel void @reduce_scan_incl_add_i64(i64 addrspace(1)* %in, i64 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i64, i64 addrspace(1)* %in, i64 %call
  %0 = load i64, i64 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i64 @__mux_sub_group_scan_inclusive_add_i64(i64 %0)
  %arrayidx2 = getelementptr inbounds i64, i64 addrspace(1)* %out, i64 %call
  store i64 %call1, i64 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_add_i64(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x i64> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4m(<vscale x 4 x i64> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call i64 @llvm.vector.reduce.add.nxv4i64(<vscale x 4 x i64> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call i64 @__mux_sub_group_scan_exclusive_add_i64(i64 [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[HEAD]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = add <vscale x 4 x i64> [[SCAN]], [[SPLAT]]
; CHECK: store <vscale x 4 x i64> [[FINAL]],
}

define spir_kernel void @reduce_scan_incl_add_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func float @__mux_sub_group_scan_inclusive_fadd_f32(float %0)
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %call1, float addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_add_f32(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4f(<vscale x 4 x float> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call float @llvm.vector.reduce.fadd.nxv4f32(float -0.0{{.*}}, <vscale x 4 x float> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call float @__mux_sub_group_scan_exclusive_fadd_f32(float [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x float> poison, float [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[HEAD]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = fadd <vscale x 4 x float> [[SCAN]], [[SPLAT]]
; CHECK: store <vscale x 4 x float> [[FINAL]],
}

define spir_kernel void @reduce_scan_incl_smin_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @__mux_sub_group_scan_inclusive_smin_i32(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_smin_i32(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_u5nxv4i(<vscale x 4 x i32> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call i32 @llvm.vector.reduce.smin.nxv4i32(<vscale x 4 x i32> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call i32 @__mux_sub_group_scan_exclusive_smin_i32(i32 [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[HEAD]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = call <vscale x 4 x i32> @llvm.smin.nxv4i32(<vscale x 4 x i32> [[SCAN]], <vscale x 4 x i32> [[SPLAT]])
; CHECK: store <vscale x 4 x i32> [[FINAL]],
}

define spir_kernel void @reduce_scan_incl_umin_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @__mux_sub_group_scan_inclusive_umin_i32(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_umin_i32(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_u5nxv4j(<vscale x 4 x i32> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call i32 @llvm.vector.reduce.umin.nxv4i32(<vscale x 4 x i32> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call i32 @__mux_sub_group_scan_exclusive_umin_i32(i32 [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[HEAD]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = call <vscale x 4 x i32> @llvm.umin.nxv4i32(<vscale x 4 x i32> [[SCAN]], <vscale x 4 x i32> [[SPLAT]])
; CHECK: store <vscale x 4 x i32> [[FINAL]],
}

define spir_kernel void @reduce_scan_incl_smax_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @__mux_sub_group_scan_inclusive_smax_i32(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_smax_i32(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_u5nxv4i(<vscale x 4 x i32> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call i32 @llvm.vector.reduce.smax.nxv4i32(<vscale x 4 x i32> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call i32 @__mux_sub_group_scan_exclusive_smax_i32(i32 [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[HEAD]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = call <vscale x 4 x i32> @llvm.smax.nxv4i32(<vscale x 4 x i32> [[SCAN]], <vscale x 4 x i32> [[SPLAT]])
; CHECK: store <vscale x 4 x i32> [[FINAL]],
}

define spir_kernel void @reduce_scan_incl_umax_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @__mux_sub_group_scan_inclusive_umax_i32(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_umax_i32(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_u5nxv4j(<vscale x 4 x i32> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call i32 @llvm.vector.reduce.umax.nxv4i32(<vscale x 4 x i32> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call i32 @__mux_sub_group_scan_exclusive_umax_i32(i32 [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[HEAD]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = call <vscale x 4 x i32> @llvm.umax.nxv4i32(<vscale x 4 x i32> [[SCAN]], <vscale x 4 x i32> [[SPLAT]])
; CHECK: store <vscale x 4 x i32> [[FINAL]],
}

define spir_kernel void @reduce_scan_incl_fmin_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func float @__mux_sub_group_scan_inclusive_fmin_f32(float %0)
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %call1, float addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_fmin_f32(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_min_u5nxv4f(<vscale x 4 x float> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call float @llvm.vector.reduce.fmin.nxv4f32(<vscale x 4 x float> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call float @__mux_sub_group_scan_exclusive_fmin_f32(float [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x float> poison, float [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[HEAD]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = call <vscale x 4 x float> @llvm.minnum.nxv4f32(<vscale x 4 x float> [[SCAN]], <vscale x 4 x float> [[SPLAT]])
; CHECK: store <vscale x 4 x float> [[FINAL]],
}

define spir_kernel void @reduce_scan_incl_fmax_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func float @__mux_sub_group_scan_inclusive_fmax_f32(float %0)
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %call1, float addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_fmax_f32(
; CHECK: [[SCAN:%.*]] = call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_max_u5nxv4f(<vscale x 4 x float> [[INPUT:%.*]])
; CHECK: [[SUM:%.*]] = call float @llvm.vector.reduce.fmax.nxv4f32(<vscale x 4 x float> [[INPUT]])
; CHECK: [[EXCL_SCAN:%.*]] = call float @__mux_sub_group_scan_exclusive_fmax_f32(float [[SUM]])
; CHECK: [[HEAD:%.*]] = insertelement <vscale x 4 x float> poison, float [[EXCL_SCAN]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[HEAD]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[FINAL:%.*]] = call <vscale x 4 x float> @llvm.maxnum.nxv4f32(<vscale x 4 x float> [[SCAN]], <vscale x 4 x float> [[SPLAT]])
; CHECK: store <vscale x 4 x float> [[FINAL]],
}
