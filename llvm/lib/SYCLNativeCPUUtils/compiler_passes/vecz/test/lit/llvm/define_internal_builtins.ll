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

; RUN: veczc -k dummy -vecz-simd-width=4 -vecz-passes=define-builtins -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @dummy(i32 addrspace(2)* %in, i32 addrspace(1)* %out) {
  %b = bitcast i32 addrspace(2)* %in to <4 x i32> addrspace(2)*
  %v = call <4 x i32> @__vecz_b_masked_load4_Dv4_jPU3AS2Dv4_jDv4_b(<4 x i32> addrspace(2)* %b, <4 x i1> zeroinitializer)
  ret void
}

declare <4 x i32> @__vecz_b_masked_load4_Dv4_jPU3AS2Dv4_jDv4_b(<4 x i32> addrspace(2)*, <4 x i1>)
; CHECK-LABEL: define <4 x i32> @__vecz_b_masked_load4_Dv4_jPU3AS2Dv4_jDv4_b(ptr addrspace(2){{.*}}, <4 x i1>{{.*}}) {
; CHECK:   %2 = call <4 x i32> @llvm.masked.load.v4i32.p2(ptr addrspace(2) %0, i32 4, <4 x i1> %1, <4 x i32> poison)
; CHECK:   ret <4 x i32> %2
; CHECK: }
