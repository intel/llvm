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
; RUN: %veczc -vecz-scalable -vecz-simd-width=4 -S < %s | %filecheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i32 @_Z16get_sub_group_idv()
declare spir_func i32 @_Z18get_sub_group_sizev()
declare spir_func i32 @_Z22get_sub_group_local_idv()
declare spir_func i32 @_Z19sub_group_broadcastij(i32, i32)

define spir_kernel void @get_sub_group_size(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call.i = tail call spir_func i32 @_Z16get_sub_group_idv()
  %conv = zext i32 %call.i to i64
  %call2 = tail call spir_func i32 @_Z18get_sub_group_sizev()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_nxv4_get_sub_group_size(
; CHECK: [[VSCALE:%.*]] = call i32 @llvm.vscale.i32()
; CHECK: [[W:%.*]] = shl i32 [[VSCALE]], 2
; CHECK: store i32 [[W]], ptr addrspace(1) {{.*}}
}

define spir_kernel void @get_sub_group_local_id(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call = tail call spir_func i32 @_Z22get_sub_group_local_idv()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %call
  store i32 %call, i32 addrspace(1)* %arrayidx, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_nxv4_get_sub_group_local_id(
; CHECK: [[LID:%.*]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; CHECK: store <vscale x 4 x i32> [[LID]], ptr addrspace(1) %out
}

define spir_kernel void @sub_group_broadcast(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call = tail call spir_func i32 @_Z22get_sub_group_local_idv()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %call
  %v = load i32, i32 addrspace(1)* %arrayidx, align 4
  %broadcast = call spir_func i32 @_Z19sub_group_broadcastij(i32 %v, i32 0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %call
  store i32 %broadcast, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_nxv4_sub_group_broadcast(
; CHECK: [[LD:%.*]] = load <vscale x 4 x i32>, ptr addrspace(1) {{%.*}}, align 4
; CHECK: [[EXT:%.*]] = extractelement <vscale x 4 x i32> [[LD]], {{(i32|i64)}} 0
; CHECK: [[INS:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[EXT]], {{(i32|i64)}} 0
; CHECK: [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[INS]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: store <vscale x 4 x i32> [[SPLAT]], ptr addrspace(1)
}

!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}
