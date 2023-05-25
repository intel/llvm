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

; RUN: veczc -vecz-scalable -vecz-simd-width=4 -S -vecz-passes=packetizer < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i64 @_Z13get_global_idj(i32)

declare spir_func i32 @_Z28sub_group_scan_inclusive_addi(i32)
declare spir_func i64 @_Z28sub_group_scan_inclusive_addl(i64)
declare spir_func float @_Z28sub_group_scan_inclusive_addf(float)

declare spir_func i32 @_Z28sub_group_scan_inclusive_mini(i32)
declare spir_func i32 @_Z28sub_group_scan_inclusive_minj(i32)
declare spir_func i32 @_Z28sub_group_scan_inclusive_maxi(i32)
declare spir_func i32 @_Z28sub_group_scan_inclusive_maxj(i32)
declare spir_func float @_Z28sub_group_scan_inclusive_minf(float)
declare spir_func float @_Z28sub_group_scan_inclusive_maxf(float)

define spir_kernel void @reduce_scan_incl_add_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z28sub_group_scan_inclusive_addi(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_add_i32(
; CHECK: call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4j(<vscale x 4 x i32> %{{.*}})
}

define spir_kernel void @reduce_scan_incl_add_i64(i64 addrspace(1)* %in, i64 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i64, i64 addrspace(1)* %in, i64 %call
  %0 = load i64, i64 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i64 @_Z28sub_group_scan_inclusive_addl(i64 %0)
  %arrayidx2 = getelementptr inbounds i64, i64 addrspace(1)* %out, i64 %call
  store i64 %call1, i64 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_add_i64(
; CHECK: call <vscale x 4 x i64> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4m(<vscale x 4 x i64> %{{.*}})
}

define spir_kernel void @reduce_scan_incl_add_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func float @_Z28sub_group_scan_inclusive_addf(float %0)
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %call1, float addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_add_f32(
; CHECK: call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_add_u5nxv4f(<vscale x 4 x float> %{{.*}})
}

define spir_kernel void @reduce_scan_incl_smin_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z28sub_group_scan_inclusive_mini(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_smin_i32(
; CHECK: call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smin_u5nxv4i(<vscale x 4 x i32> %{{.*}})
}

define spir_kernel void @reduce_scan_incl_umin_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z28sub_group_scan_inclusive_minj(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_umin_i32(
; CHECK: call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umin_u5nxv4j(<vscale x 4 x i32> %{{.*}})
}

define spir_kernel void @reduce_scan_incl_smax_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z28sub_group_scan_inclusive_maxi(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_smax_i32(
; CHECK: call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_smax_u5nxv4i(<vscale x 4 x i32> %{{.*}})
}

define spir_kernel void @reduce_scan_incl_umax_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z28sub_group_scan_inclusive_maxj(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_umax_i32(
; CHECK: call <vscale x 4 x i32> @__vecz_b_sub_group_scan_inclusive_umax_u5nxv4j(<vscale x 4 x i32> %{{.*}})
}

define spir_kernel void @reduce_scan_incl_fmin_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func float @_Z28sub_group_scan_inclusive_minf(float %0)
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %call1, float addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_fmin_f32(
; CHECK: call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_min_u5nxv4f(<vscale x 4 x float> %{{.*}})
}

define spir_kernel void @reduce_scan_incl_fmax_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call1 = tail call spir_func float @_Z28sub_group_scan_inclusive_maxf(float %0)
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %call1, float addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: @__vecz_nxv4_reduce_scan_incl_fmax_f32(
; CHECK: call <vscale x 4 x float> @__vecz_b_sub_group_scan_inclusive_max_u5nxv4f(<vscale x 4 x float> %{{.*}})
}

!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
