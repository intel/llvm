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

; RUN: veczc -vecz-simd-width=4 -S < %s | FileCheck %s 

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i32 @__mux_get_sub_group_id()
declare spir_func i32 @__mux_get_sub_group_size()
declare spir_func i32 @__mux_get_sub_group_local_id()
declare spir_func i32 @__mux_sub_group_broadcast_i32(i32, i32)
declare spir_func i64 @__mux_get_global_id(i32)
declare spir_func i1 @__mux_sub_group_any_i1(i1)

define spir_kernel void @get_sub_group_size(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call.i = tail call spir_func i32 @__mux_get_sub_group_id()
  %conv = zext i32 %call.i to i64
  %call2 = tail call spir_func i32 @__mux_get_sub_group_size()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %conv
  store i32 %call2, i32 addrspace(1)* %arrayidx, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_v4_get_sub_group_size(
; CHECK: [[RED:%.*]] = call i32 @__mux_sub_group_reduce_add_i32(i32 4)
; CHECK: store i32 [[RED]], ptr addrspace(1) {{.*}}
}

define spir_kernel void @get_sub_group_local_id(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call = tail call spir_func i32 @__mux_get_sub_group_local_id()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %call
  store i32 %call, i32 addrspace(1)* %arrayidx, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_v4_get_sub_group_local_id(
; CHECK: %call = tail call spir_func i32 @__mux_get_sub_group_local_id()
; CHECK: [[MUL:%.*]] = shl i32 %call, 2
; CHECK: [[SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[MUL]], i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x i32> [[SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK: [[ID:%.*]] = or {{(disjoint )?}}<4 x i32> [[SPLAT]], <i32 0, i32 1, i32 2, i32 3>
; CHECK: [[EXT:%.*]] = sext i32 %call to i64
; CHECK: %arrayidx = getelementptr i32, ptr addrspace(1) %out, i64 [[EXT]]
; CHECK: store <4 x i32> [[ID]], ptr addrspace(1) %arrayidx
}

define spir_kernel void @sub_group_broadcast(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call = tail call spir_func i32 @__mux_get_sub_group_local_id()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %call
  %v = load i32, i32 addrspace(1)* %arrayidx, align 4
  %broadcast = call spir_func i32 @__mux_sub_group_broadcast_i32(i32 %v, i32 0)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %call
  store i32 %broadcast, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_v4_sub_group_broadcast(
; CHECK: [[LD:%.*]] = load <4 x i32>, ptr addrspace(1) {{%.*}}, align 4
; CHECK: [[EXT:%.*]] = extractelement <4 x i32> [[LD]], i64 0
; CHECK: [[BDCAST:%.*]] = call spir_func i32 @__mux_sub_group_broadcast_i32(i32 [[EXT]], i32 0)
; CHECK: [[HEAD:%.*]] = insertelement <4 x i32> poison, i32 [[BDCAST]], i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x i32> [[HEAD]], <4 x i32> {{(undef|poison)}}, <4 x i32> zeroinitializer
; CHECK: store <4 x i32> [[SPLAT]], ptr addrspace(1)
}

define spir_kernel void @sub_group_broadcast_wider_than_vf(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
  %call = tail call spir_func i32 @__mux_get_sub_group_local_id()
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %call
  %v = load i32, i32 addrspace(1)* %arrayidx, align 4
  %broadcast = call spir_func i32 @__mux_sub_group_broadcast_i32(i32 %v, i32 6)
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %call
  store i32 %broadcast, i32 addrspace(1)* %arrayidx2, align 4
  ret void
; CHECK-LABEL: define spir_kernel void @__vecz_v4_sub_group_broadcast_wider_than_vf(
; CHECK: [[LD:%.*]] = load <4 x i32>, ptr addrspace(1) {{%.*}}, align 4
; The sixth sub-group member is the (6 % 4 ==) 2nd vector group member
; CHECK: [[EXT:%.*]] = extractelement <4 x i32> [[LD]], i64 2
; CHECK: [[BDCAST:%.*]] = call spir_func i32 @__mux_sub_group_broadcast_i32(i32 [[EXT]], i32 1)
; CHECK: [[HEAD:%.*]] = insertelement <4 x i32> poison, i32 [[BDCAST]], i64 0
; CHECK: [[SPLAT:%.*]] = shufflevector <4 x i32> [[HEAD]], <4 x i32> {{(undef|poison)}}, <4 x i32> zeroinitializer
; CHECK: store <4 x i32> [[SPLAT]], ptr addrspace(1)
}

; This used to crash as packetizing get_sub_group_local_id produces a Constant, which we weren't expecting.
define spir_kernel void @regression_sub_group_local_id(i32 addrspace(1)* %in, <4 x i32> addrspace(1)* %xy, i32 addrspace(1)* %out) {
entry:
  %call = tail call spir_func i64 @__mux_get_global_id(i32 0)
  %call1 = tail call spir_func i32 @__mux_get_sub_group_local_id()
  %0 = shl i64 %call, 32
  %idxprom = ashr exact i64 %0, 32
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %xy, i64 %idxprom
  %1 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidx, align 16
  %2 = insertelement <4 x i32> %1, i32 %call1, i64 0
  %3 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arrayidx, i64 0, i64 0
  store i32 %call1, i32 addrspace(1)* %3, align 16
  %call2 = tail call spir_func i32 @__mux_get_sub_group_id()
  %4 = insertelement <4 x i32> %2, i32 %call2, i64 1
  store <4 x i32> %4, <4 x i32> addrspace(1)* %arrayidx, align 16
  %arrayidx6 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom
  %5 = load i32, i32 addrspace(1)* %arrayidx6, align 4
  %6 = icmp ne i32 %5, 0
  %call7 = tail call spir_func i1 @__mux_sub_group_any_i1(i1 %6)
  %7 = sext i1 %call7 to i32
  %arrayidx9 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %7, i32 addrspace(1)* %arrayidx9, align 4
  ret void
}
