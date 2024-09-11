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

; REQUIRES: llvm-13+
; RUN: veczc -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i32 @__mux_get_sub_group_id()
declare spir_func i32 @__mux_get_sub_group_size()
declare spir_func i32 @__mux_get_sub_group_local_id()
declare spir_func i32 @__mux_sub_group_broadcast_i32(i32, i32)

define spir_kernel void @get_sub_group_size(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call.i = tail call spir_func i32 @__mux_get_sub_group_id()
  %conv = zext i32 %call.i to i64
  %call2 = tail call spir_func i32 @__mux_get_sub_group_size()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_nxv4_get_sub_group_size(
; CHECK: [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK: [[W:%.*]] = shl i32 [[VSCALE]], 2
; CHECK: [[RED:%.*]] = call i32 @__mux_sub_group_reduce_add_i32(i32 [[W]])
; CHECK: store i32 [[RED]], ptr addrspace(1) {{.*}}
}

define spir_kernel void @get_sub_group_local_id(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call = tail call spir_func i32 @__mux_get_sub_group_local_id()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %call
  store i32 %call, i32 addrspace(1)* %arrayidx, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_nxv4_get_sub_group_local_id(
; CHECK: %call = tail call spir_func i32 @__mux_get_sub_group_local_id()
; CHECK: [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK: [[SHL:%.*]] = shl i32 %1, 2
; CHECK: [[MUL:%.*]] = mul i32 %call, [[SHL]]
; CHECK: [[SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[MUL]], i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: [[STEPVEC:%.*]] = call <vscale x 4 x i32> @llvm.{{(experimental\.)?}}stepvector.nxv4i32()
; CHECK: [[LID:%.*]] = add <vscale x 4 x i32> [[SPLAT]], [[STEPVEC]]
; CHECK: [[EXT:%.*]] = sext i32 %call to i64
; CHECK: %arrayidx = getelementptr i32, ptr addrspace(1) %out, i64 [[EXT]]
; CHECK: store <vscale x 4 x i32> [[LID]], ptr addrspace(1) %arrayidx
}

define spir_kernel void @sub_group_broadcast(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call = tail call spir_func i32 @__mux_get_sub_group_local_id()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %call
  %v = load i32, i32 addrspace(1)* %arrayidx, align 4
  %broadcast = call spir_func i32 @__mux_sub_group_broadcast_i32(i32 %v, i32 0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %call
  store i32 %broadcast, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_nxv4_sub_group_broadcast(
; CHECK: [[LD:%.*]] = load <vscale x 4 x i32>, ptr addrspace(1) {{%.*}}, align 4
; CHECK: [[EXT:%.*]] = extractelement <vscale x 4 x i32> [[LD]], {{(i32|i64)}} 0
; CHECK: [[BDCAST:%.*]] = call spir_func i32 @__mux_sub_group_broadcast_i32(i32 [[EXT]], i32 0)
; CHECK: [[INS:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[BDCAST]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[INS]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: store <vscale x 4 x i32> [[SPLAT]], ptr addrspace(1)
}

!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
