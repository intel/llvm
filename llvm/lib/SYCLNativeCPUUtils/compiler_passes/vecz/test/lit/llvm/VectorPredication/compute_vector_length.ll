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
; RUN: veczc -k get_sub_group_size -vecz-simd-width=2 -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK-F2
; RUN: veczc -k get_sub_group_size -vecz-scalable -vecz-simd-width=4 -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK-S4

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i32 @__mux_get_sub_group_id()
declare spir_func i32 @__mux_get_sub_group_size()

define spir_kernel void @get_sub_group_size(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call.i = tail call spir_func i32 @__mux_get_sub_group_id()
  %conv = zext i32 %call.i to i64
  %call2 = tail call spir_func i32 @__mux_get_sub_group_size()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Makes sure the vector length is properly computed and substituted for get_sub_group_size()

; CHECK-F2-LABEL: define spir_kernel void @__vecz_v2_vp_get_sub_group_size(
; CHECK-F2: [[ID:%.*]] = call i64 @__mux_get_local_id(i32 0)
; CHECK-F2: [[SZ:%.*]] = call i64 @__mux_get_local_size(i32 0)
; CHECK-F2: [[WL:%.*]] = sub {{.*}} i64 [[SZ]], [[ID]]
; CHECK-F2: [[VL0:%.*]] = call i64 @llvm.umin.i64(i64 [[WL]], i64 2)
; CHECK-F2: [[VL1:%.*]] = trunc {{(nuw )?(nsw )?}}i64 [[VL0]] to i32
; CHECK-F2: [[RED:%.*]] = call i32 @__mux_sub_group_reduce_add_i32(i32 [[VL1]])
; CHECK-F2: store i32 [[RED]], ptr addrspace(1) {{.*}}

; CHECK-S4-LABEL: define spir_kernel void @__vecz_nxv4_vp_get_sub_group_size(
; CHECK-S4: [[ID:%.*]] = call i64 @__mux_get_local_id(i32 0)
; CHECK-S4: [[SZ:%.*]] = call i64 @__mux_get_local_size(i32 0)
; CHECK-S4: [[WL:%.*]] = sub {{.*}} i64 [[SZ]], [[ID]]
; CHECK-S4: [[VF0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-S4: [[VF1:%.*]] = shl i64 [[VF0]], 2
; CHECK-S4: [[VL0:%.*]] = call i64 @llvm.umin.i64(i64 [[WL]], i64 [[VF1]])
; CHECK-S4: [[VL1:%.*]] = trunc {{(nuw )?(nsw )?}}i64 [[VL0]] to i32
; CHECK-S4: [[RED:%.*]] = call i32 @__mux_sub_group_reduce_add_i32(i32 [[VL1]])
; CHECK-S4: store i32 [[RED]], ptr addrspace(1) {{.*}}
