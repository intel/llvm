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

; RUN: %veczc -k test -w 4 -S < %s | %filecheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

declare spir_func i32 @get_local_id(i32);
declare spir_func i32 @get_global_id(i32);

define spir_kernel void @test(i32 addrspace(1)* %in) {
entry:
  %lid = call i32 @get_local_id(i32 0)
  %and = and i32 %lid, 1
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if, label %merge

if:
  %lid1 = call i32 @get_local_id(i32 1)
  %cmp1 = icmp eq i32 %lid1, 0
  br i1 %cmp1, label %deeper_if, label %deeper_merge

deeper_if:
  br label %deeper_merge

deeper_merge:
  %load = load i32, i32 addrspace(1)* %in
  %gid = call i32 @get_global_id(i32 0)
  %slot = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %gid
  store i32 %load, i32 addrspace(1)* %slot
  br label %merge

merge:
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_test
; CHECK: %[[LOAD:.+]] = load i32, ptr addrspace(1) %in
; CHECK: %[[SPLAT_IN:.+]] = insertelement <4 x i32> {{poison|undef}}, i32 %[[LOAD]], {{(i32|i64)}} 0
; CHECK: %[[SPLAT:.+]] = shufflevector <4 x i32> %[[SPLAT_IN]], <4 x i32> {{poison|undef}}, <4 x i32> zeroinitializer
; CHECK: call void @__vecz_b_masked_store4_Dv4_ju3ptrU3AS1Dv4_b(<4 x i32> %[[SPLAT]], ptr addrspace(1){{( nonnull)? %.*}}, <4 x i1> %{{.+}})
