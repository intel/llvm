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

; RUN: veczc -k constant_index -vecz-simd-width=4 -vecz-passes=packetizer -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @constant_index(<4 x i32>* %in, <4 x i32>* %out) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32>* %in, i64 %call
  %0 = load <4 x i32>, <4 x i32>* %arrayidx
  %arrayidx2 = getelementptr inbounds <4 x i32>, <4 x i32>* %out, i64 %call
  %vecins = insertelement <4 x i32> %0, i32 42, i32 2
  store <4 x i32> %vecins, <4 x i32>* %arrayidx2
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_constant_index

; A single wide load
; CHECK: %[[INTO:.+]] = load <16 x i32>, ptr %

; No interleaved loads
; CHECK-NOT: call <4 x i32> @__vecz_b_interleaved_load4_Dv4_ju3ptr

; Insert constant elements into the widened vector:
; CHECK: %[[INS0:.+]] = insertelement <16 x i32> %[[INTO]], i32 42, i32 2
; CHECK: %[[INS1:.+]] = insertelement <16 x i32> %[[INS0]], i32 42, i32 6
; CHECK: %[[INS2:.+]] = insertelement <16 x i32> %[[INS1]], i32 42, i32 10
; CHECK: %[[INS3:.+]] = insertelement <16 x i32> %[[INS2]], i32 42, i32 14

; No shuffles..
; CHECK-NOT: shufflevector

; We should have one widened store
; CHECK: store <16 x i32> %[[INS3]]
; CHECK: ret void
