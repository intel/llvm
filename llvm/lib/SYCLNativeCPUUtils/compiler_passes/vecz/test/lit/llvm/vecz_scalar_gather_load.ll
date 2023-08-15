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

; RUN: veczc -k vecz_scalar_gather_load -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_group_id(i32)

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_local_id(i32)

; Function Attrs: convergent nounwind
define spir_kernel void @vecz_scalar_gather_load(i32 addrspace(1)* %row_indices, i32 addrspace(1)* %row_blocks, float addrspace(1)* %result) {
entry:
  %call1 = call i64 @__mux_get_group_id(i32 0)
  %call2 = call i64 @__mux_get_local_id(i32 0)
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %row_blocks, i64 %call1
  %load1 = load i32, i32 addrspace(1)* %arrayidx1, align 4
  %add1 = add i64 %call1, 1
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %row_blocks, i64 %add1
  %load2 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  br label %for.cond

for.cond:                                       ; preds = %entry, %for.inc
  %storemerge = phi i32 [ %load1, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ult i32 %storemerge, %load2
  br i1 %cmp1, label %if.then1, label %for.end

if.then1:                                       ; preds = %for.cond
  %storemerge.zext = zext i32 %storemerge to i64
  %gep1 = getelementptr inbounds i32, i32 addrspace(1)* %row_indices, i64 %storemerge.zext
  %load3 = load i32, i32 addrspace(1)* %gep1, align 4
  %sub1 = sub i32 %load3, %load1
  %gep2 = getelementptr inbounds i32, i32 addrspace(1)* %row_indices, i64 %storemerge.zext
  %load4 = load i32, i32 addrspace(1)* %gep2, align 4
  %sub2 = sub i32 %load4, %load1
  %cmp2 = icmp ugt i32 %sub2, %sub1
  br i1 %cmp2, label %if.then2, label %if.else2

if.then2:                                       ; preds = %if.then1
  %sub1.zext = zext i32 %sub1 to i64
  %gep3 = getelementptr inbounds float, float addrspace (1)* %result, i64 %sub1.zext
  %load5 = load float, float addrspace(1)* %gep3, align 4
  br label %if.else2

if.else2:                                        ; preds = %if.then1, %if.then2
  %ret = phi float [ %load5, %if.then2 ], [ 0.000000e+00, %if.then1 ]
  %cmp3 = icmp eq i64 %call2, 0
  br i1 %cmp3, label %if.then3, label %for.inc

if.then3:                                       ; preds = %if.else2
  %gep4 = getelementptr inbounds float, float addrspace(1)* %result, i64 %call2
  store float %ret, float addrspace(1)* %gep4, align 4
  br label %for.inc

for.inc:                                       ; preds = %if.then3, %if.else2
  %inc = add i32 %storemerge, 1
  br label %for.cond

for.end:                                        ; preds = %for.cond
  ret void
}

; The purpose of this test is to ensure we don't generate a masked load for a
; load from a uniform address, even where it is in a divergent control path.
; It used to be the case that such a load would become a masked load during
; control flow conversion, thefore causing it to become a varying load due to
; the varying mask. However, since the introduction of the Mask Varying
; attribute, it is possible to support a Uniform load with a Varying mask, so
; it is no longer necessary to mark all loads in divergent paths as Varying.
; The somewhat circuitous upshot of this is that the load no longer gets a mask
; at all, since it was previously only considered to be in a divergent path on
; account of another Mask Varying load!

; CHECK: spir_kernel void @__vecz_v4_vecz_scalar_gather_load

; This load depends only on the uniform loop iterator
; CHECK: if.then1:
; CHECK: %[[IND:.+]] = phi i32
; CHECK: %[[ZIND:.+]] = zext i32 %[[IND]] to i64
; CHECK: %[[GEP1:.+]] = getelementptr inbounds i32, ptr addrspace(1) %row_indices, i64 %[[ZIND]]
; CHECK: %{{.+}} = load i32, ptr addrspace(1) %[[GEP1]]

; This load depends only on other uniform loads
; CHECK: if.then2:
; CHECK-NOT: declare float @__vecz_b_masked_gather_load4_
; CHECK-NOT: declare float @__vecz_b_masked_load4_
; CHECK: %[[GEP2:.+]] = getelementptr inbounds float, ptr addrspace(1) %result
; CHECK: %{{.+}} = load float, ptr addrspace(1) %[[GEP2]]

; The store instruction is definitely in a divergent path, however, so needs a mask.
; CHECK: if.then3:
; CHECK: call void @__vecz_b_masked_store4_f
