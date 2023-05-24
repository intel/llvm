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

; REQUIRES: llvm-15+
; RUN: %veczc -k test -vecz-simd-width=4 -vecz-passes=gep-elim -S < %s | %filecheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%struct.mystruct = type { [2 x i32], ptr }

; Function Attrs: norecurse nounwind
define spir_kernel void @test(ptr addrspace(1) nocapture writeonly align 4 %output) {
entry:
  %foo = alloca [4 x %struct.mystruct], align 4
  %call = tail call spir_func i32 @_Z13get_global_idj(i32 0)
  store i32 20, ptr %foo, align 4
  %arrayidx4 = getelementptr inbounds [2 x i32], ptr %foo, i32 0, i32 1
  store i32 22, ptr %arrayidx4, align 4
  %y31 = getelementptr inbounds %struct.mystruct, ptr %foo, i32 0, i32 1
  store ptr %foo, ptr %y31, align 4
  %mul = shl nuw nsw i32 %call, 2
  store i32 1, ptr %foo, align 4
  %0 = load ptr, ptr %y31, align 4
  %1 = load i32, ptr %0, align 4
  %add98 = add nsw i32 %mul, %1
  %arrayidx117 = getelementptr inbounds i32, ptr addrspace(1) %output, i32 %mul
  store i32 %add98, ptr addrspace(1) %arrayidx117, align 4
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK: define spir_kernel void @__vecz_v4_test(

; Make sure all three GEPs are retained
; CHECK: %arrayidx4 = getelementptr inbounds [2 x i32], ptr %foo, i32 0, i32 1
; CHECK: %y31 = getelementptr inbounds %struct.mystruct, ptr %foo, i32 0, i32 1
; CHECK: %arrayidx117 = getelementptr inbounds i32, ptr addrspace(1) %output, i32 %mul
; CHECK: ret void
