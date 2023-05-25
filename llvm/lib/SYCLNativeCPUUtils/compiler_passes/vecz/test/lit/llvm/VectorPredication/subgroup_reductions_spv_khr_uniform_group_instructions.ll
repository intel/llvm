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

declare spir_func i32 @_Z20sub_group_reduce_muli(i32)
declare spir_func i64 @_Z20sub_group_reduce_mull(i64)
declare spir_func float @_Z20sub_group_reduce_mulf(float)

declare spir_func i32 @_Z20sub_group_reduce_andj(i32)
declare spir_func i32 @_Z19sub_group_reduce_ori(i32)
declare spir_func i64 @_Z20sub_group_reduce_xorl(i64)

declare spir_func i1 @_Z28sub_group_reduce_logical_andb(i1)
declare spir_func i1 @_Z27sub_group_reduce_logical_orb(i1)
declare spir_func i1 @_Z28sub_group_reduce_logical_xorb(i1)

; CHECK-LABEL: @__vecz_v4_vp_reduce_mul_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i32> {{%.*}}, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.mul.v4i32(<4 x i32> [[I]])
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_mul_i32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z20sub_group_reduce_muli(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %conv
  store i32 %call2, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-LABEL: @__vecz_v4_vp_reduce_mul_i64(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i64> {{%.*}}, <4 x i64> <i64 1, i64 1, i64 1, i64 1>
; CHECK: [[R:%.*]] = call i64 @llvm.vector.reduce.mul.v4i64(<4 x i64> [[I]])
; CHECK: store i64 [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_mul_i64(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i64, ptr addrspace(1) %in, i64 %call
  %0 = load i64, ptr addrspace(1) %arrayidx, align 4
  %call2 = tail call spir_func i64 @_Z20sub_group_reduce_mull(i64 %0)
  %arrayidx3 = getelementptr inbounds i64, ptr addrspace(1) %out, i64 %conv
  store i64 %call2, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-LABEL: @__vecz_v4_vp_reduce_mul_f32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x float> {{%.*}}, <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
; CHECK: [[R:%.*]] = call float @llvm.vector.reduce.fmul.v4f32(float 1.000000e+00, <4 x float> [[I]])
; CHECK: store float [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_mul_f32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %in, i64 %call
  %0 = load float, ptr addrspace(1) %arrayidx, align 4
  %call2 = tail call spir_func float @_Z20sub_group_reduce_mulf(float %0)
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %conv
  store float %call2, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-LABEL: @__vecz_v4_vp_reduce_and_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i32> {{%.*}}, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1> 
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.and.v4i32(<4 x i32> [[I]])
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_and_i32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z20sub_group_reduce_andj(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %conv
  store i32 %call2, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-LABEL: @__vecz_v4_vp_reduce_or_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i32> {{%.*}}, <4 x i32> zeroinitializer
; CHECK: [[R:%.*]] = call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> [[I]])
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_or_i32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %call2 = tail call spir_func i32 @_Z19sub_group_reduce_ori(i32 %0)
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %conv
  store i32 %call2, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-LABEL: @__vecz_v4_vp_reduce_xor_i32(
; CHECK: [[SI:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, {{(i32|i64)}} 0
; CHECK: [[S:%.*]] = shufflevector <4 x i32> [[SI]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[C:%.*]] = icmp ugt <4 x i32> [[S]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[I:%.*]] = select <4 x i1> [[C]], <4 x i64> {{%.*}}, <4 x i64> zeroinitializer
; CHECK: [[R:%.*]] = call i64 @llvm.vector.reduce.xor.v4i64(<4 x i64> [[I]])
; CHECK: store i64 [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_xor_i32(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %call
  %0 = load i64, ptr addrspace(1) %arrayidx, align 4
  %call2 = tail call spir_func i64 @_Z20sub_group_reduce_xorl(i64 %0)
  %arrayidx3 = getelementptr inbounds i64, ptr addrspace(1) %out, i64 %conv
  store i64 %call2, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-LABEL: @__vecz_v4_vp_reduce_logical_and(
; This doesn't generate a reduction intrinsic...
; CHECK: [[T:%.*]] = icmp eq i4 {{%.*}}, -1
; CHECK: [[R:%.*]] = zext i1 [[T]] to i32
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_logical_and(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %1 = trunc i32 %0 to i1
  %call2 = tail call spir_func i1 @_Z28sub_group_reduce_logical_andb(i1 %1)
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %conv
  %zext = zext i1 %call2 to i32
  store i32 %zext, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-LABEL: @__vecz_v4_vp_reduce_logical_or(
; CHECK: [[T:%.*]] = icmp ne i4 {{%.*}}, 0
; CHECK: [[R:%.*]] = zext i1 [[T]] to i32
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_logical_or(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %1 = trunc i32 %0 to i1
  %call2 = tail call spir_func i1 @_Z27sub_group_reduce_logical_orb(i1 %1)
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %conv
  %zext = zext i1 %call2 to i32
  store i32 %zext, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-LABEL: @__vecz_v4_vp_reduce_logical_xor(
; CHECK: [[X:%.*]] = call i4 @llvm.ctpop.i4(i4 {{%.*}})
; CHECK: [[T:%.*]] = and i4 [[X]], 1
; CHECK: [[R:%.*]] = zext i4 [[T]] to i32
; CHECK: store i32 [[R]], ptr addrspace(1) {{%.*}}, align 4
define spir_kernel void @reduce_logical_xor(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i32 @_Z16get_sub_group_idv() #6
  %conv = zext i32 %call1 to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %1 = trunc i32 %0 to i1
  %call2 = tail call spir_func i1 @_Z28sub_group_reduce_logical_xorb(i1 %1)
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %conv
  %zext = zext i1 %call2 to i32
  store i32 %zext, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
