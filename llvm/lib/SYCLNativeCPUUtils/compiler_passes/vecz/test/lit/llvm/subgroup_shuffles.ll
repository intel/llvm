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

; RUN: veczc -w 4 -vecz-passes=packetizer -S \
; RUN:   --pass-remarks-missed=vecz < %s 2>&1 | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Could not packetize sub-group shuffle %shuffle
define spir_kernel void @kernel1(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %size = call i32 @__mux_get_sub_group_size()
  %size_minus_1 = sub i32 %size, 1
  %arrayidx.in = getelementptr inbounds i64, ptr addrspace(1) %in, i64 %gid
  %val = load i64, ptr addrspace(1) %arrayidx.in, align 8
  %shuffle = call i64 @__mux_sub_group_shuffle_i64(i64 %val, i32 %size_minus_1)
  %arrayidx.out = getelementptr inbounds i64, ptr addrspace(1) %out, i64 %gid
  store i64 %shuffle, ptr addrspace(1) %arrayidx.out, align 8
  ret void
}

; CHECK: Could not packetize sub-group shuffle %shuffle_up
define spir_kernel void @kernel2(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds float, ptr addrspace(1) %in, i64 %gid
  %val = load float, ptr addrspace(1) %arrayidx.in, align 8
  %shuffle_up = call float @__mux_sub_group_shuffle_up_f32(float %val, float %val, i32 1)
  %arrayidx.out = getelementptr inbounds float, ptr addrspace(1) %out, i64 %gid
  store float %shuffle_up, ptr addrspace(1) %arrayidx.out, align 8
  ret void
}

; CHECK: Could not packetize sub-group shuffle %shuffle_down
define spir_kernel void @kernel3(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds i8, ptr addrspace(1) %in, i64 %gid
  %val = load i8, ptr addrspace(1) %arrayidx.in, align 8
  %shuffle_down = call i8 @__mux_sub_group_shuffle_down_i8(i8 %val, i8 %val, i32 1)
  %arrayidx.out = getelementptr inbounds i8, ptr addrspace(1) %out, i64 %gid
  store i8 %shuffle_down, ptr addrspace(1) %arrayidx.out, align 8
  ret void
}

; CHECK: Could not packetize sub-group shuffle %shuffle_xor
define spir_kernel void @kernel4(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds half, ptr addrspace(1) %in, i64 %gid
  %val = load half, ptr addrspace(1) %arrayidx.in, align 8
  %shuffle_xor = call half @__mux_sub_group_shuffle_xor_f16(half %val, i32 -1)
  %arrayidx.out = getelementptr inbounds half, ptr addrspace(1) %out, i64 %gid
  store half %shuffle_xor, ptr addrspace(1) %arrayidx.out, align 8
  ret void
}

declare i64 @__mux_get_global_id(i32)

declare i32 @__mux_get_sub_group_size()

declare i64 @__mux_sub_group_shuffle_i64(i64 %val, i32 %lid)

declare half @__mux_sub_group_shuffle_xor_f16(half %val, i32 %xor_val)

declare i8 @__mux_sub_group_shuffle_down_i8(i8 %curr, i8 %next, i32 %delta)

declare float @__mux_sub_group_shuffle_up_f32(float %prev, float %curr, i32 %delta)
