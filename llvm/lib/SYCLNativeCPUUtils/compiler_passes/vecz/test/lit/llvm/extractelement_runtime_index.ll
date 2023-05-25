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

; RUN: veczc -k extract_runtime_index -vecz-simd-width=4 -vecz-passes=scalarize -vecz-choices=FullScalarization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: nounwind
define spir_kernel void @extract_runtime_index(<4 x float> addrspace(1)* %in, i32 %x, float addrspace(1)* %out) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 4
  %vecext = extractelement <4 x float> %0, i32 %x
  %arrayidx1 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %vecext, float addrspace(1)* %arrayidx1, align 4
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_extract_runtime_index
; CHECK: load float, {{(ptr|float)}}
; CHECK: load float, {{(ptr|float)}}
; CHECK: load float, {{(ptr|float)}}
; CHECK: load float, {{(ptr|float)}}
; CHECK: icmp eq i32 0, %x
; CHECK: select i1
; CHECK: icmp eq i32 1, %x
; CHECK: select i1
; CHECK: icmp eq i32 2, %x
; CHECK: select i1
; CHECK: icmp eq i32 3, %x
; CHECK: select i1
; CHECK: store float
; CHECK: ret void
