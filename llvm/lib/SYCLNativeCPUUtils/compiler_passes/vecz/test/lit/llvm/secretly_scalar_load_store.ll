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

; RUN: veczc -w 4 -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

declare i32 @__mux_get_global_id(i32);

define spir_kernel void @test(i32 addrspace(1)* %in) {
entry:
  %gid = call i32 @__mux_get_global_id(i32 0)
  %and = and i32 %gid, 1
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if, label %early_ret

early_ret:
; just to prevent ROSCC from sticking its oar in
  %gid1 = call i32 @__mux_get_global_id(i32 1)
  ret void

if:
  %single_load = load i32, i32 addrspace(1)* %in
  %single_add = add i32 %single_load, 42
  store i32 %single_add, i32 addrspace(1)* %in

  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_test
; CHECK: %[[BITCAST:.*]] = bitcast <4 x i1> %cmp{{[0-9]*}} to i4
; CHECK: %[[MASK:.*]] = icmp ne i4 %[[BITCAST]], 0
; CHECK: %[[single_load:single_load[0-9]*]] = call i32 @__vecz_b_masked_load4_ju3ptrU3AS1b(ptr addrspace(1) %in, i1 %[[MASK]])
; CHECK: %[[single_add:single_add[0-9]*]] = add i32 %[[single_load]], 42
; CHECK: call void @__vecz_b_masked_store4_ju3ptrU3AS1b(i32 %[[single_add]], ptr addrspace(1) %in, i1 %[[MASK]])
