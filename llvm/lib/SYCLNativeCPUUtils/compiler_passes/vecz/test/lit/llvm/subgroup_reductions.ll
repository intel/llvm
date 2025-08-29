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

; RUN: veczc -w 4 -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i64 @__mux_get_global_id(i32)
declare spir_func i32 @__mux_get_sub_group_id()

declare spir_func i1 @__mux_sub_group_all_i1(i1)
declare spir_func i1 @__mux_sub_group_any_i1(i1)

declare spir_func i32 @__mux_sub_group_reduce_add_i32(i32)
declare spir_func i64 @__mux_sub_group_reduce_add_i64(i64)
declare spir_func float @__mux_sub_group_reduce_fadd_f32(float)
declare spir_func i32 @__mux_sub_group_reduce_smin_i32(i32)
declare spir_func i32 @__mux_sub_group_reduce_umin_i32(i32)
declare spir_func i32 @__mux_sub_group_reduce_smax_i32(i32)
declare spir_func i32 @__mux_sub_group_reduce_umax_i32(i32)
declare spir_func float @__mux_sub_group_reduce_fmin_f32(float)
declare spir_func float @__mux_sub_group_reduce_fmax_f32(float)

define spir_kernel void @reduce_all_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %1 = icmp ne i32 %0, 0
  %call2 = tail call spir_func i1 @__mux_sub_group_all_i1(i1 %1)
  %2 = sext i1 %call2 to i32
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_all_i32(
; CHECK: [[T2:%.*]] = icmp eq <4 x i32> %{{.*}}, zeroinitializer

; CHECK: [[T3:%.*]] = bitcast <4 x i1> [[T2]] to i4
; CHECK: [[R:%.*]] = icmp eq i4 [[T3]], 0
; CHECK: %call2 = tail call spir_func i1 @__mux_sub_group_all_i1(i1 [[R]])
; CHECK: [[EXT:%.*]] = sext i1 %call2 to i32
; CHECK: store i32 [[EXT]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_any_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %1 = icmp ne i32 %0, 0
  %call2 = tail call spir_func i1 @__mux_sub_group_any_i1(i1 %1)
  %2 = sext i1 %call2 to i32
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_any_i32(
; CHECK: [[T2:%.*]] = icmp ne <4 x i32> %{{.*}}, zeroinitializer

; CHECK: [[T3:%.*]] = bitcast <4 x i1> [[T2]] to i4
; CHECK: [[R:%.*]] = icmp ne i4 [[T3]], 0
; CHECK: %call2 = tail call spir_func i1 @__mux_sub_group_any_i1(i1 [[R]])
; CHECK: [[EXT:%.*]] = sext i1 %call2 to i32
; CHECK: store i32 [[EXT]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_add_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @__mux_sub_group_reduce_add_i32(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_add_i32(
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %{{.*}})
; CHECK: %call2 = tail call spir_func i32 @__mux_sub_group_reduce_add_i32(i32 [[R]])
; CHECK: store i32 %call2, ptr addrspace(1) {{%.*}}, align 4
}

; Given we've checked a "full" expanded reduction sequence above for LLVM < 13,
; reduce duplicate CHECKs by assuming all reductions work orthogonally.

define spir_kernel void @reduce_add_i32_uniform(i32 addrspace(1)* %out, i32 %n) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_sub_group_reduce_add_i32(i32 %n)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %call1, i32 addrspace(1)* %arrayidx, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_add_i32_uniform(
; LLVM is clever enough to optimize this reduction, but not when it's an
; intrinsic. LLVM 10 does the shift-left in a vector, LLVMs 11 and 12 do it in
; scalar.
; CHECK: [[CALL:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> {{%.*}})
; CHECK: %call1 = tail call spir_func i32 @__mux_sub_group_reduce_add_i32(i32 [[CALL]])
; CHECK: [[INS:%.*]] = insertelement <4 x i32> poison, i32 %call1, {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x i32> [[INS]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: store <4 x i32> [[SPLAT]],
}

define spir_kernel void @reduce_add_i64(i64 addrspace(1)* %in, i64 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i64, i64 addrspace(1)* %in, i64 %call
  %0 = load i64, i64 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i64 @__mux_sub_group_reduce_add_i64(i64 %0)
  %arrayidx3 = getelementptr inbounds i64, i64 addrspace(1)* %out, i64 %conv
  store i64 %call2, i64 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_add_i64(
; CHECK: [[R:%.*]] = call i64 @llvm.vector.reduce.add.v4i64(<4 x i64> %{{.*}})
; CHECK: %call2 = tail call spir_func i64 @__mux_sub_group_reduce_add_i64(i64 [[R]])
; CHECK: store i64 %call2, ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_add_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func float @__mux_sub_group_reduce_fadd_f32(float %0)
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %conv
  store float %call2, float addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_add_f32(
; CHECK: [[R:%.*]] = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %{{.*}})
; CHECK: %call2 = tail call spir_func float @__mux_sub_group_reduce_fadd_f32(float [[R]])
; CHECK: store float %call2, ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_smin_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @__mux_sub_group_reduce_smin_i32(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_smin_i32(
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.smin.v4i32(<4 x i32> %{{.*}})
; CHECK: %call2 = tail call spir_func i32 @__mux_sub_group_reduce_smin_i32(i32 [[R]])
; CHECK: store i32 %call2, ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_umin_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @__mux_sub_group_reduce_umin_i32(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_umin_i32(
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> %{{.*}})
; CHECK: %call2 = tail call spir_func i32 @__mux_sub_group_reduce_umin_i32(i32 [[R]])
; CHECK: store i32 %call2, ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_smax_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @__mux_sub_group_reduce_smax_i32(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_smax_i32(
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %{{.*}})
; CHECK: %call2 = tail call spir_func i32 @__mux_sub_group_reduce_smax_i32(i32 [[R]])
; CHECK: store i32 %call2, ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_umax_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @__mux_sub_group_reduce_umax_i32(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_umax_i32(
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> %{{.*}})
; CHECK: %call2 = tail call spir_func i32 @__mux_sub_group_reduce_umax_i32(i32 [[R]])
; CHECK: store i32 %call2, ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_fmin_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func float @__mux_sub_group_reduce_fmin_f32(float %0)
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %conv
  store float %call2, float addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_fmin_f32(
; CHECK: [[R:%.*]] = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> %{{.*}})
; CHECK: %call2 = tail call spir_func float @__mux_sub_group_reduce_fmin_f32(float [[R]])
; CHECK: store float %call2, ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_fmax_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_id() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func float @__mux_sub_group_reduce_fmax_f32(float %0)
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %conv
  store float %call2, float addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_reduce_fmax_f32(
; CHECK: [[R:%.*]] = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %{{.*}})
; CHECK: %call2 = tail call spir_func float @__mux_sub_group_reduce_fmax_f32(float [[R]])
; CHECK: store float %call2, ptr addrspace(1) {{%.*}}, align 4
}

!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
