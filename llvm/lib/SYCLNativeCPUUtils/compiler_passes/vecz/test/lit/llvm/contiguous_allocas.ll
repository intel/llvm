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

; RUN: %veczc -k test -vecz-simd-width=4 -vecz-auto -vecz-choices=FullScalarization -S < %s | %filecheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@entry_test_alloca.lm = internal unnamed_addr addrspace(3) constant [16 x <2 x float>] undef, align 8

define spir_kernel void @test(<2 x float> addrspace(1)* nocapture readonly %in, <2 x float> addrspace(1)* nocapture %out, i32 %offset) local_unnamed_addr {
entry:
  %a.sroa.0 = alloca <2 x float>, align 8
  %b.sroa.2 = alloca <2 x float>, align 8
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %call1 = tail call spir_func i64 @_Z12get_local_idj(i32 0)
  %a.sroa.0.0..sroa_cast = bitcast <2 x float>* %a.sroa.0 to i8*
  %b.sroa.2.0..sroa_cast = bitcast <2 x float>* %b.sroa.2 to i8*
  %arrayidx2 = getelementptr inbounds [16 x <2 x float>], [16 x <2 x float>] addrspace(3)* @entry_test_alloca.lm, i64 0, i64 %call1
  %0 = load <2 x float>, <2 x float> addrspace(3)* %arrayidx2, align 8
  %conv = sext i32 %offset to i64
  %add = add i64 %call1, %conv
  %arrayidx4 = getelementptr inbounds [16 x <2 x float>], [16 x <2 x float>] addrspace(3)* @entry_test_alloca.lm, i64 0, i64 %add
  %1 = load <2 x float>, <2 x float> addrspace(3)* %arrayidx4, align 8
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup10
  %mul.le.le = fmul <2 x float> %a.sroa.0.0.a.sroa.0.0.a.sroa.0.0., %b.sroa.2.0.b.sroa.2.0.b.sroa.2.8.
  %arrayidx17 = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %out, i64 %call
  store <2 x float> %mul.le.le, <2 x float> addrspace(1)* %arrayidx17, align 8
  ret void

for.body:                                         ; preds = %for.cond.cleanup10, %entry
  %i.038 = phi i32 [ 0, %entry ], [ %inc15, %for.cond.cleanup10 ]
  store volatile <2 x float> %0, <2 x float>* %a.sroa.0, align 8
  store volatile <2 x float> %1, <2 x float>* %b.sroa.2, align 8
  br label %for.body11

for.cond.cleanup10:                               ; preds = %for.body11
  %inc15 = add nuw nsw i32 %i.038, 1
  %cmp = icmp ult i32 %inc15, 16
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.body11:                                       ; preds = %for.body11, %for.body
  %i6.037 = phi i32 [ 0, %for.body ], [ %inc, %for.body11 ]
  %a.sroa.0.0.a.sroa.0.0.a.sroa.0.0. = load volatile <2 x float>, <2 x float>* %a.sroa.0, align 8
  %b.sroa.2.0.b.sroa.2.0.b.sroa.2.8. = load volatile <2 x float>, <2 x float>* %b.sroa.2, align 8
  %inc = add nuw nsw i32 %i6.037, 1
  %cmp8 = icmp ult i32 %inc, 16
  br i1 %cmp8, label %for.body11, label %for.cond.cleanup10
}

declare spir_func i64 @_Z13get_global_idj(i32) local_unnamed_addr
declare spir_func i64 @_Z12get_local_idj(i32) local_unnamed_addr

; Check that all the allocas come before anything else
; CHECK: define spir_kernel void @__vecz_v4_test(
; CHECK-NEXT: entry:
; CHECK-NEXT: %a.sroa.{{[0-9]+}} = alloca <8 x float>, align 16
; CHECK-NEXT: %b.sroa.{{[0-9]+}} = alloca <8 x float>, align 16
