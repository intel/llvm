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

; RUN: veczc -k scan_fact -vecz-passes=cfg-convert -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@scan_fact.temp = internal addrspace(3) global [16 x i32] undef, align 4

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32) #0

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_local_id(i32) #0

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_local_size(i32) #0

; Function Attrs: convergent nounwind
define spir_kernel void @scan_fact(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #1 {
entry:
  %call = call i64 @__mux_get_local_id(i32 0) #3
  %call1 = call i64 @__mux_get_global_id(i32 0) #3
  %call2 = call i64 @__mux_get_local_size(i32 0) #3
  %mul = shl i64 %call1, 1
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %mul
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %mul3 = shl i64 %call, 1
  %arrayidx4 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %mul3
  store i32 %0, i32 addrspace(3)* %arrayidx4, align 4
  %mul5 = shl i64 %call1, 1
  %add = or i64 %mul5, 1
  %arrayidx6 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %add
  %1 = load i32, i32 addrspace(1)* %arrayidx6, align 4
  %mul7 = shl i64 %call, 1
  %add8 = or i64 %mul7, 1
  %arrayidx9 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %add8
  store i32 %1, i32 addrspace(3)* %arrayidx9, align 4
  %mul10 = shl i64 %call, 1
  %add11 = or i64 %mul10, 1
  %arrayidx12 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %add11
  %2 = load i32, i32 addrspace(3)* %arrayidx12, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %storemerge = phi i64 [ 1, %entry ], [ %mul29, %for.inc ]
  %mul13 = shl i64 %call2, 1
  %cmp = icmp ult i64 %storemerge, %mul13
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @__mux_work_group_barrier(i32 1, i32 1, i32 272) #4
  %mul14 = shl i64 %call, 1
  %mul15 = mul i64 %storemerge, %mul14
  %mul16 = shl i64 %call2, 1
  %cmp17 = icmp ult i64 %mul15, %mul16
  br i1 %cmp17, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %mul18 = mul i64 %storemerge, 2
  %add19 = add i64 %mul15, -1
  %sub = add i64 %add19, %mul18
  %arrayidx20 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %sub
  %3 = load i32, i32 addrspace(3)* %arrayidx20, align 4
  %add21 = add i64 %mul15, -1
  %sub22 = add i64 %add21, %storemerge
  %arrayidx23 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %sub22
  %4 = load i32, i32 addrspace(3)* %arrayidx23, align 4
  %mul24 = mul nsw i32 %4, %3
  %mul25 = mul i64 %storemerge, 2
  %add26 = add i64 %mul15, -1
  %sub27 = add i64 %add26, %mul25
  %arrayidx28 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %sub27
  store i32 %mul24, i32 addrspace(3)* %arrayidx28, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %mul29 = shl i64 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %cmp30 = icmp eq i64 %call, 0
  br i1 %cmp30, label %if.then31, label %if.end35

if.then31:                                        ; preds = %for.end
  %mul32 = mul i64 %call2, 2
  %sub33 = add i64 %mul32, -1
  %arrayidx34 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %sub33
  store i32 1, i32 addrspace(3)* %arrayidx34, align 4
  br label %if.end35

if.end35:                                         ; preds = %if.then31, %for.end
  br label %for.cond37

for.cond37:                                       ; preds = %for.inc62, %if.end35
  %storemerge1 = phi i64 [ %call2, %if.end35 ], [ %shr, %for.inc62 ]
  %cmp38 = icmp eq i64 %storemerge1, 0
  call void @__mux_work_group_barrier(i32 1, i32 1, i32 272) #4
  %mul64 = shl i64 %call, 1
  br i1 %cmp38, label %for.end63, label %for.body39

for.body39:                                       ; preds = %for.cond37
  %mul42 = mul i64 %storemerge1, %mul64
  %mul43 = shl i64 %call2, 1
  %cmp44 = icmp ult i64 %mul42, %mul43
  br i1 %cmp44, label %if.then45, label %for.inc62

if.then45:                                        ; preds = %for.body39
  %add46 = add i64 %mul42, -1
  %sub47 = add i64 %add46, %storemerge1
  %arrayidx48 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %sub47
  %5 = load i32, i32 addrspace(3)* %arrayidx48, align 4
  %mul49 = mul i64 %storemerge1, 2
  %add50 = add i64 %mul42, -1
  %sub51 = add i64 %add50, %mul49
  %arrayidx52 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %sub51
  %6 = load i32, i32 addrspace(3)* %arrayidx52, align 4
  %add53 = add i64 %mul42, -1
  %sub54 = add i64 %add53, %storemerge1
  %arrayidx55 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %sub54
  store i32 %6, i32 addrspace(3)* %arrayidx55, align 4
  %mul56 = mul nsw i32 %6, %5
  %mul57 = mul i64 %storemerge1, 2
  %add58 = add i64 %mul42, -1
  %sub59 = add i64 %add58, %mul57
  %arrayidx60 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %sub59
  store i32 %mul56, i32 addrspace(3)* %arrayidx60, align 4
  br label %for.inc62

for.inc62:                                        ; preds = %if.then45, %for.body39
  %shr = lshr i64 %storemerge1, 1
  br label %for.cond37

for.end63:                                        ; preds = %for.cond37
  %add65 = or i64 %mul64, 1
  %arrayidx66 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %add65
  %7 = load i32, i32 addrspace(3)* %arrayidx66, align 4
  %mul67 = shl i64 %call1, 1
  %arrayidx68 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %mul67
  store i32 %7, i32 addrspace(1)* %arrayidx68, align 4
  %sub69 = add i64 %call2, -1
  %cmp70 = icmp eq i64 %call, %sub69
  br i1 %cmp70, label %if.then71, label %if.else

if.then71:                                        ; preds = %for.end63
  %mul72 = shl i64 %call, 1
  %add73 = or i64 %mul72, 1
  %arrayidx74 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %add73
  %8 = load i32, i32 addrspace(3)* %arrayidx74, align 4
  %mul75 = mul nsw i32 %8, %2
  %mul76 = shl i64 %call1, 1
  %add77 = or i64 %mul76, 1
  %arrayidx78 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %add77
  store i32 %mul75, i32 addrspace(1)* %arrayidx78, align 4
  br label %if.end85

if.else:                                          ; preds = %for.end63
  %mul79 = mul i64 %call, 2
  %add80 = add i64 %mul79, 2
  %arrayidx81 = getelementptr inbounds [16 x i32], [16 x i32] addrspace(3)* @scan_fact.temp, i64 0, i64 %add80
  %9 = load i32, i32 addrspace(3)* %arrayidx81, align 4
  %mul82 = shl i64 %call1, 1
  %add83 = or i64 %mul82, 1
  %arrayidx84 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %add83
  store i32 %9, i32 addrspace(1)* %arrayidx84, align 4
  br label %if.end85

if.end85:                                         ; preds = %if.else, %if.then71
  ret void
}

declare void @__mux_work_group_barrier(i32, i32, i32)

; The purpose of this test is to make sure we simply manage to vectorize this
; test. We would previously not because a phi node of a uniform loop has an
; incoming value from a divergent block, but all the incoming values of the
; phi node are the same.
; We would thus previously consider the phi node varying and that would make
; the loop divergent, with a barrier in it.

; CHECK: spir_kernel void @__vecz_v4_scan_fact
