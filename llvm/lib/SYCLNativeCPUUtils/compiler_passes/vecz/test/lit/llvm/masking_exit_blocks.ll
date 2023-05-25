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

; RUN: veczc -k test -vecz-simd-width=4 -vecz-passes=cfg-convert,packetizer -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@.str = private unnamed_addr addrspace(2) constant [18 x i8] c"Doing stuff, yay!\00", align 1

define spir_kernel void @test(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %add = add i64 %call, 1
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %add
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  br label %entry.1

entry.1:                                          ; preds = %entry
  %add1 = add i64 %call, 1
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %add1
  store i32 %0, i32 addrspace(1)* %arrayidx2, align 4
  %cmp = icmp eq i64 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry.1
  %call3 = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([18 x i8], [18 x i8] addrspace(2)* @.str, i64 0, i64 0))
  br label %if.end

if.end:                                           ; preds = %if.then, %entry.1
  %arrayidx4 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %1 = load i32, i32 addrspace(1)* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  br label %if.end1

if.end1:                                          ; preds = %if.end
  store i32 %1, i32 addrspace(1)* %arrayidx5, align 4
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

declare extern_weak spir_func i32 @printf(i8 addrspace(2)*, ...)

; CHECK: define spir_kernel void @__vecz_v4_test

; Check if the divergent block is masked correctly
; CHECK: @__vecz_b_masked_printf_u3ptrU3AS2b
; CHECK: @__vecz_b_masked_printf_u3ptrU3AS2b
; CHECK: @__vecz_b_masked_printf_u3ptrU3AS2b
; CHECK: @__vecz_b_masked_printf_u3ptrU3AS2b

; Check if the exit block is not masked
; CHECK: load <4 x i32>
; CHECK: store <4 x i32>

; CHECK: ret void
