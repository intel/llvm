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

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @constant_index(<4 x i32>* %in, i32* %inval, <4 x i32>* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32>* %in, i64 %call
  %0 = load <4 x i32>, <4 x i32>* %arrayidx
  %arrayidx2 = getelementptr inbounds i32, i32* %inval, i64 %call
  %ldval = load i32, i32* %arrayidx2
  %arrayidx3 = getelementptr inbounds <4 x i32>, <4 x i32>* %out, i64 %call
  %vecins = insertelement <4 x i32> %0, i32 %ldval, i32 2
  store <4 x i32> %vecins, <4 x i32>* %arrayidx3
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_constant_index

; A single wide load
; CHECK: %[[INTO:.+]] = load <16 x i32>, ptr %

; The vectorized element load:
; CHECK: %[[ELTS:.+]] = load <4 x i32>, ptr %

; No interleaved loads
; CHECK-NOT: call <4 x i32> @__vecz_b_interleaved_load4_Dv4_ju3ptr

; Insert elements turned into shufflevectors
; CHECK: %[[WIDE:.+]] = shufflevector <4 x i32> %[[ELTS]], <4 x i32> undef, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 3>
; CHECK: %[[INS:.+]] = shufflevector <16 x i32> %[[WIDE]], <16 x i32> %[[INTO]], <16 x i32> <i32 16, i32 17, i32 2, i32 19, i32 20, i32 21, i32 6, i32 23, i32 24, i32 25, i32 10, i32 27, i32 28, i32 29, i32 14, i32 31>

; No more shuffles..
; CHECK-NOT: shufflevector

; We should have one widened store
; CHECK: store <16 x i32> %[[INS]]
; CHECK: ret void
