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

; RUN: veczc -w 4 -vecz-choices=VectorPredication -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func i32 @_Z16get_sub_group_idv()

declare spir_func i32 @_Z13sub_group_alli(i32)
declare spir_func i32 @_Z13sub_group_anyi(i32)

declare spir_func i32 @_Z20sub_group_reduce_addi(i32)
declare spir_func i64 @_Z20sub_group_reduce_addl(i64)
declare spir_func float @_Z20sub_group_reduce_addf(float)
declare spir_func i32 @_Z20sub_group_reduce_mini(i32)
declare spir_func i32 @_Z20sub_group_reduce_minj(i32)
declare spir_func i32 @_Z20sub_group_reduce_maxi(i32)
declare spir_func i32 @_Z20sub_group_reduce_maxj(i32)
declare spir_func float @_Z20sub_group_reduce_minf(float)
declare spir_func float @_Z20sub_group_reduce_maxf(float)

define spir_kernel void @reduce_all_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z13sub_group_alli(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_all_i32(
; CHECK: [[T2:%.*]] = icmp ne <4 x i32> {{%.*}}, zeroinitializer
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ult <4 x i32> [[S]], <i32 1, i32 2, i32 3, i32 4>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i1> [[T2]]
; CHECK: [[T3:%.*]] = bitcast <4 x i1> [[I]] to i4
; CHECK: [[R:%.*]] = icmp eq i4 [[T3]], -1
; CHECK: [[EXT:%.*]] = sext i1 [[R]] to i32
; CHECK: store i32 [[EXT]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_any_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z13sub_group_anyi(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_any_i32(
; CHECK: [[T2:%.*]] = icmp ne <4 x i32> {{%.*}}, zeroinitializer
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i1> [[T2]], <4 x i1> zeroinitializer
; CHECK: [[T3:%.*]] = bitcast <4 x i1> [[I]] to i4
; CHECK: [[R:%.*]] = icmp ne i4 [[T3]], 0
; CHECK: [[EXT:%.*]] = sext i1 [[R]] to i32
; CHECK: store i32 [[EXT]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_add_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z20sub_group_reduce_addi(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_add_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i32> {{%.*}}, <4 x i32> zeroinitializer
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[I]])
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_add_i64(i64 addrspace(1)* %in, i64 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i64, i64 addrspace(1)* %in, i64 %call
  %0 = load i64, i64 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i64 @_Z20sub_group_reduce_addl(i64 %0)
  %arrayidx3 = getelementptr inbounds i64, i64 addrspace(1)* %out, i64 %conv
  store i64 %call2, i64 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_add_i64(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i64> {{%.*}}, <4 x i64> zeroinitializer
; CHECK: [[R:%.*]] = call i64 @llvm.vector.reduce.add.v4i64(<4 x i64> [[I]])
; CHECK: store i64 [[R]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_add_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func float @_Z20sub_group_reduce_addf(float %0)
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %conv
  store float %call2, float addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_add_f32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x float> {{%.*}}, <4 x float> <float -0.000000e+00, float -0.000000e+00,
; CHECK: [[R:%.*]] = call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> [[I]])
; CHECK: store float [[R]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_smin_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z20sub_group_reduce_mini(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_smin_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i32> {{%.*}}, <4 x i32> <i32 2147483647, i32 2147483647, 
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.smin.v4i32(<4 x i32> [[I]])
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_umin_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z20sub_group_reduce_minj(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_umin_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i32> {{%.*}}, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> [[I]])
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_smax_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z20sub_group_reduce_maxi(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_smax_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i32> {{%.*}}, <4 x i32> <i32 -2147483648, i32 -2147483648, 
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> [[I]])
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_umax_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z20sub_group_reduce_maxj(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_umax_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i32> {{%.*}}, <4 x i32> zeroinitializer
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> [[I]])
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_fmin_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func float @_Z20sub_group_reduce_minf(float %0)
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %conv
  store float %call2, float addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_fmin_f32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x float> {{%.*}}, <4 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000,
; CHECK: [[R:%.*]] = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> [[I]])
; CHECK: store float [[R]], ptr addrspace(1) {{%.*}}, align 4
}

define spir_kernel void @reduce_fmax_f32(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %in, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %call2 = tail call spir_func float @_Z20sub_group_reduce_maxf(float %0)
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %conv
  store float %call2, float addrspace(1)* %arrayidx3, align 4
  ret void
; CHECK-LABEL: @__vecz_v4_vp_reduce_fmax_f32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x float> {{%.*}}, <4 x float> <float 0xFFF8000000000000, float 0xFFF8000000000000,
; CHECK: [[R:%.*]] = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> [[I]])
; CHECK: store float [[R]], ptr addrspace(1) {{%.*}}, align 4
}

!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
