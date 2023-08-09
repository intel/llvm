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

; RUN: veczc -w 4 -S -vecz-passes=packetizer < %s | FileCheck %s

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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_add_i32(
; CHECK: call <4 x i32> @__vecz_b_sub_group_scan_inclusive_add_Dv4_j(<4 x i32> %{{.*}})
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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_add_i64(
; CHECK: call <4 x i64> @__vecz_b_sub_group_scan_inclusive_add_Dv4_m(<4 x i64> %{{.*}})
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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_add_f32(
; CHECK: call <4 x float> @__vecz_b_sub_group_scan_inclusive_add_Dv4_f(<4 x float> %{{.*}})
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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_smin_i32(
; CHECK: call <4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_Dv4_i(<4 x i32> %{{.*}})
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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_umin_i32(
; CHECK: call <4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_Dv4_j(<4 x i32> %{{.*}})
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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_smax_i32(
; CHECK: call <4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_Dv4_i(<4 x i32> %{{.*}})
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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_umax_i32(
; CHECK: call <4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_Dv4_j(<4 x i32> %{{.*}})
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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_fmin_f32(
; CHECK: call <4 x float> @__vecz_b_sub_group_scan_inclusive_min_Dv4_f(<4 x float> %{{.*}})
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
; CHECK-LABEL: @__vecz_v4_reduce_scan_incl_fmax_f32(
; CHECK: call <4 x float> @__vecz_b_sub_group_scan_inclusive_max_Dv4_f(<4 x float> %{{.*}})
}
