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

; RUN: %veczc -k test < %s

; This test ensures that VECZ does not crash during control flow conversion due
; to a missing exit mask. As such, we need only verify that the return code from
; veczc is 0, and FileCheck is not required. See CA-3117 for details.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(i32 addrspace(1)* %out, i32 %n) {
entry:
  %call = tail call spir_func i32 @_Z13get_global_idj(i32 0)
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body.preheader, label %if.end.thread

for.body.preheader:
  %cmp2 = icmp sgt i32 %n, 1
  %0 = and i32 %call, 1
  %cmp3 = icmp eq i32 %0, 0
  br i1 %cmp2, label %if.end2, label %if.else

if.end.thread:
  %cmp4 = icmp eq i32 %call, 0
  br i1 %cmp4, label %if.end, label %for.cond.preheader

if.else:
  br i1 %cmp3, label %if.end, label %for.body

for.cond.preheader:
  %cmp5 = icmp sgt i32 %n, 1
  br i1 %cmp5, label %for.body, label %if.end

for.body:
  br i1 0, label %if.end, label %for.body

if.end:
  %div = sdiv i32 %call, 2
  br label %if.end2

if.end2:
  %ret = phi i32 [ 0, %for.body.preheader ], [ %div, %if.end ]
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 0
  store i32 %ret, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

declare spir_func i32 @_Z13get_global_idj(i32)

declare spir_func i32 @_Z3maxii(i32, i32)
