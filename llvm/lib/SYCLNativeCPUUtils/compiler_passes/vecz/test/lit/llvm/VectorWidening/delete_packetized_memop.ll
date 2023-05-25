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

; RUN: veczc -k memop_loop_dep -vecz-simd-width=4 -vecz-passes=builtin-inlining,packetizer -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-s128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @memop_loop_dep(i32 addrspace(1)* %in, i32 addrspace(1)* %out, i32 %i, i32 %e) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.addr.0 = phi i32 [ %i, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.addr.0, %e
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call1 = call spir_func <4 x i32> @_Z6vload4mPKU3AS1i(i64 %call, i32 addrspace(1)* %in)
  call spir_func void @_Z7vstore4Dv4_imPU3AS1i(<4 x i32> %call1, i64 %call, i32 addrspace(1)* %out)
  %0 = extractelement <4 x i32> %call1, i64 0
  %tobool = icmp ne i32 %0, 0
  %tobool2 = icmp eq i64 %call, 0
  %or.cond = and i1 %tobool2, %tobool
  br i1 %or.cond, label %while.cond, label %for.inc

while.cond:                                       ; preds = %while.cond, %for.body
  %tobool3 = icmp eq i64 %call, 0
  br i1 %tobool3, label %for.inc, label %while.cond

for.inc:                                          ; preds = %for.body, %while.cond
  %inc = add nsw i32 %i.addr.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

declare spir_func <4 x i32> @_Z6vload4mPKU3AS1i(i64, i32 addrspace(1)*)

declare spir_func void @_Z7vstore4Dv4_imPU3AS1i(<4 x i32>, i64, i32 addrspace(1)*)

; CHECK: define spir_kernel void @__vecz_v4_memop_loop_dep

; Vector widening results in a single load
; CHECK: load <16 x i32>
; CHECK-NOT: call {{.*}}i32 @__vecz_b_interleaved_load4_ju3ptrU3AS1

; CHECK: ret void

; Check if the declaration is missing as well
; CHECK-NOT: @__vecz_b_interleaved_load4_ju3ptrU3AS1
