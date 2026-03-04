; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: veczc -k widen_shufflevector -vecz-simd-width=2 -vecz-passes=packetizer -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32) #1

; Function Attrs: nounwind
define spir_kernel void @widen_shufflevector(<2 x float> addrspace(1)* %a, <2 x float> addrspace(1)* %b, <4 x float> addrspace(1)* %out) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %arrayidxa = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %a, i64 %call
  %arrayidxb = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %b, i64 %call
  %la = load <2 x float>, <2 x float> addrspace(1)* %arrayidxa, align 4
  %lb = load <2 x float>, <2 x float> addrspace(1)* %arrayidxb, align 4
  %shuffle = shufflevector <2 x float> %la, <2 x float> %lb, <4 x i32> <i32 0, i32 3, i32 1, i32 2>
  %arrayidx1 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %shuffle, <4 x float> addrspace(1)* %arrayidx1, align 1
  ret void
}

; CHECK: define spir_kernel void @__vecz_v2_widen_shufflevector
; CHECK: %[[LDA:.+]] = load <4 x float>, ptr addrspace(1) %
; CHECK: %[[LDB:.+]] = load <4 x float>, ptr addrspace(1) %
; CHECK: %[[SHF:.+]] = shufflevector <4 x float> %[[LDA]], <4 x float> %[[LDB]], <8 x i32> <i32 0, i32 5, i32 1, i32 4, i32 2, i32 7, i32 3, i32 6>
; CHECK: store <8 x float> %[[SHF]], ptr addrspace(1) %
; CHECK: ret void
