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

; RUN: veczc -k test_isnanDv4_f -vecz-simd-width=4 -vecz-passes=builtin-inlining,packetizer -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)
declare spir_func <4 x i32> @_Z5isnanDv4_f(<4 x float>)

define spir_kernel void @test_isnanDv4_f(<4 x float> addrspace(1)* %in, <4 x i32> addrspace(1)* %out) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 16
  %call1 = call spir_func <4 x i32> @_Z5isnanDv4_f(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out, i64 %call
  store <4 x i32> %call1, <4 x i32> addrspace(1)* %arrayidx2, align 16
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_test_isnanDv4_f
; CHECK: and <16 x i32>
; CHECK: icmp eq <16 x i32>
; CHECK: and <16 x i32>
; CHECK: icmp sgt <16 x i32>
; CHECK: and <16 x i1>
; CHECK: sext <16 x i1>
; CHECK: ret void
