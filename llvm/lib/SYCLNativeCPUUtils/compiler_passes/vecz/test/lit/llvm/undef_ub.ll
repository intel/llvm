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

; RUN: veczc -k test -w 4 -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind readonly
declare spir_func i32 @_Z14get_local_sizej(i32) #2

; Function Attrs: convergent nounwind readonly
declare spir_func i32 @_Z12get_local_idj(i32) #2

; Function Attrs: convergent nounwind
define spir_kernel void @test() #0 {
entry:
  %call8 = call spir_func i32 @_Z12get_local_idj(i32 0) #3
  %arrayidx = getelementptr inbounds i8, i8 addrspace(1)* undef, i32 %call8
  %0 = load i8, i8 addrspace(1)* %arrayidx, align 1
  %conv9 = uitofp i8 %0 to float
  %phitmp = fptoui float %conv9 to i8
  %arrayidx16 = getelementptr inbounds i8, i8 addrspace(1)* undef, i32 %call8
  store i8 %phitmp, i8 addrspace(1)* %arrayidx16, align 1
  ret void
}

; The "undefs" in the above IR should "optimize" to a trap call and an unreachable
; terminator instruction.
; CHECK: define spir_kernel void @__vecz_v4_test
; On LLVM 13+ there's no such trap: the UB is just that the function returns early.
; CHECK: ret void
