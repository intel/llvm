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

; RUN: veczc -k test -vecz-simd-width=4 -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%struct.T = type { i32, i8, float, i64 }

; Function Attrs: nounwind
define spir_kernel void @test(%struct.T addrspace(1)* %in, %struct.T addrspace(1)* %out, i32 addrspace(1)* %offsets) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %offsets, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %conv = sext i32 %0 to i64
  %add = add i64 %conv, %call
  %c = getelementptr inbounds %struct.T, %struct.T addrspace(1)* %in, i64 %add, i32 2
  %1 = load float, float addrspace(1)* %c, align 8
  %c3 = getelementptr inbounds %struct.T, %struct.T addrspace(1)* %out, i64 %add, i32 2
  store float %1, float addrspace(1)* %c3, align 8
  ret void
}

declare i64 @__mux_get_global_id(i32)

; Check if we can packetize GEPs on structs
; Note that we only need to packetize the non-uniform operands..
; CHECK: define spir_kernel void @__vecz_v4_test
; CHECK: getelementptr %struct.T, ptr addrspace(1) %{{.+}}, <4 x i64> %{{.+}}, i32 2
; CHECK: getelementptr %struct.T, ptr addrspace(1) %{{.+}}, <4 x i64> %{{.+}}, i32 2
