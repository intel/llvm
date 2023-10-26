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

; RUN: veczc -w 4 -vecz-passes=packetizer,verify -S \
; RUN:   --pass-remarks-missed=vecz < %s 2>&1 | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Could not packetize sub-group shuffle %shuffle_up
define spir_kernel void @kernel(ptr %in, ptr %out) {
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx.in = getelementptr inbounds float, ptr %in, i64 %gid
  %val = load float, ptr %arrayidx.in, align 8
  %shuffle_up = call float @__mux_sub_group_shuffle_up_f32(float %val, float %val, i32 1)
  %arrayidx.out = getelementptr inbounds float, ptr %out, i64 %gid
  store float %shuffle_up, ptr %arrayidx.out, align 8
  ret void
}

declare i64 @__mux_get_global_id(i32)

declare float @__mux_sub_group_shuffle_up_f32(float %prev, float %curr, i32 %delta)
