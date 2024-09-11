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

; RUN: veczc -k extract_constant_index -vecz-simd-width=4 -vecz-choices=FullScalarization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @extract_constant_index(<4 x float> addrspace(1)* %in, i32 %x, float addrspace(1)* %out) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 4
  %vecext = extractelement <4 x float> %0, i32 0;
  %arrayidx1 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %vecext, float addrspace(1)* %arrayidx1, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32) #1

; CHECK: define spir_kernel void @__vecz_v4_extract_constant_index
; CHECK: call <4 x float> @__vecz_b_interleaved_load4_4_Dv4
; CHECK: getelementptr float
; CHECK: store <4 x float>
; CHECK: ret void
