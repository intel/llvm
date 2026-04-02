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

; RUN: veczc -k uniform_reassociation -vecz-simd-width=4 -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @uniform_reassociation(i32 addrspace(1)* noalias %a, i32 addrspace(1)* noalias %b, i32 addrspace(1)* noalias %d) #0 {
entry:
  %x = call i64 @__mux_get_global_id(i32 0) #2
  %y = call i64 @__mux_get_global_id(i32 1) #2
  %a_gep = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %x
  %b_gep = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %x
  %c_gep = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %y
  %varying1 = load i32, i32 addrspace(1)* %a_gep
  %varying2 = load i32, i32 addrspace(1)* %b_gep
  %uniform = load i32, i32 addrspace(1)* %c_gep
  %vu = add i32 %varying1, %uniform
  %vvu = add i32 %varying2, %vu
  %d_gep = getelementptr inbounds i32, i32 addrspace(1)* %d, i64 %x
  store i32 %vvu, i32 addrspace(1)* %d_gep
  ret void
}

declare i64 @__mux_get_global_id(i32)

; This test checks that a sum of a varying value with two uniform values
; gets re-associated from Varying + (Varying + Uniform)
; to (Varying + Varying) + Uniform
; CHECK: define spir_kernel void @__vecz_v4_uniform_reassociation

; CHECK: %[[VARYING1:.+]] = load <4 x i32>
; CHECK: %[[VARYING2:.+]] = load <4 x i32>

; The splat of the uniform value
; CHECK: %uniform = load
; CHECK: %[[SPLATINS:.+]] = insertelement <4 x i32> poison, i32 %uniform, {{(i32|i64)}} 0
; CHECK: %[[SPLAT:.+]] = shufflevector <4 x i32> %[[SPLATINS]], <4 x i32> poison, <4 x i32> zeroinitializer

; Ensure the two varyings are added together directly
; CHECK: %[[REASSOC:.+]] = add <4 x i32> %[[VARYING1]], %[[VARYING2]]
; CHECK: %[[VVU:.+]] = add <4 x i32> %{{.*}}, %[[SPLAT]]
; CHECK: store <4 x i32> %[[VVU]], ptr addrspace(1) %{{.+}}
