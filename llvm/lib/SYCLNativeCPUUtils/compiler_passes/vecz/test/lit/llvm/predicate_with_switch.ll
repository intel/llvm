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

; RUN: veczc -k predicate_with_switch -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_local_id(i32)

declare i64 @__mux_get_global_id(i32)

@predicate_with_switch.tmpIn = internal addrspace(3) global [16 x i32] undef, align 4

define spir_kernel void @predicate_with_switch(i32 addrspace(1)* %A, i32 addrspace(1)* %B) #0 {
entry:
  %call = call i64 @__mux_get_local_id(i32 0) #2
  %call1 = call i64 @__mux_get_global_id(i32 0) #2
  switch i64 %call, label %if.end [
    i64 0, label %return
    i64 200, label %return
  ]

if.end:
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %A, i64 %call1
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @predicate_with_switch.tmpIn, i64 0, i64 %call
  store i32 %0, i32 addrspace(3)* %arrayidx3, align 4
  %sub = add i64 %call, -1
  %arrayidx4 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @predicate_with_switch.tmpIn, i64 0, i64 %sub
  %1 = load i32, i32 addrspace(3)* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32, i32 addrspace(1)* %B, i64 %call1
  store i32 %1, i32 addrspace(1)* %arrayidx5, align 4
  br label %return

return:
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_predicate_with_switch

; We should use masked stores
; CHECK: vecz_b_masked_store4
; CHECK: vecz_b_masked_store4

; We should *not* have unconditional stores
; CHECK-NOT: store <4 x i32>
; CHECK-NOT: store <4 x i32>
