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

; RUN: %veczc -k noreduce -vecz-choices=PacketizeUniform -vecz-simd-width=4 -S < %s | %filecheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z12get_local_idj(i32)
declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func i64 @_Z14get_local_sizej(i32)

; Function Attrs: nounwind
define spir_kernel void @reduce(i32 addrspace(3)* %in, i32 addrspace(3)* %out) {
entry:
  %call = call spir_func i64 @_Z12get_local_idj(i32 0)
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %storemerge = phi i32 [ 1, %entry ], [ %mul6, %for.inc ]
  %conv = zext i32 %storemerge to i64
  %call1 = call spir_func i64 @_Z14get_local_sizej(i32 0)
  %cmp = icmp ult i64 %conv, %call1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %mul = mul i32 %storemerge, 3
  %conv3 = zext i32 %mul to i64
  %0 = icmp eq i32 %mul, 0
  %1 = select i1 %0, i64 1, i64 %conv3
  %rem = urem i64 %call, %1
  %cmp4 = icmp eq i64 %rem, 0
  br i1 %cmp4, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32 addrspace(3)* %out, i64 %call
  store i32 5, i32 addrspace(3)* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %mul6 = shl i32 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @noreduce(i32 addrspace(3)* %in, i32 addrspace(3)* %out) {
entry:
  %call = call spir_func i64 @_Z12get_local_idj(i32 0)
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %storemerge = phi i32 [ 1, %entry ], [ %mul, %for.inc ]
  %conv = zext i32 %storemerge to i64
  %call1 = call spir_func i64 @_Z14get_local_sizej(i32 0)
  %cmp = icmp ult i64 %conv, %call1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = icmp eq i32 %storemerge, 0
  %1 = select i1 %0, i32 1, i32 %storemerge
  %rem = urem i32 3, %1
  %cmp3 = icmp eq i32 %rem, 0
  br i1 %cmp3, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32 addrspace(3)* %out, i64 %call
  store i32 5, i32 addrspace(3)* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %mul = shl i32 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @noreduce2(i32 addrspace(3)* %in, i32 addrspace(3)* %out) {
entry:
  %call = call spir_func i64 @_Z12get_local_idj(i32 0)
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %storemerge = phi i32 [ 1, %entry ], [ %mul, %for.inc ]
  %conv = zext i32 %storemerge to i64
  %call1 = call spir_func i64 @_Z14get_local_sizej(i32 0)
  %cmp = icmp ult i64 %conv, %call1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = icmp eq i32 %storemerge, 0
  %1 = select i1 %0, i32 1, i32 %storemerge
  %rem = urem i32 3, %1
  %cmp3 = icmp eq i32 %rem, 0
  br i1 %cmp3, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %idxprom = zext i32 %storemerge to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(3)* %out, i64 %idxprom
  store i32 5, i32 addrspace(3)* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %mul = shl i32 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @conditional(i32 addrspace(1)* %in, i32 addrspace(1)* %out) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %rem1 = and i32 %0, 1
  %tobool = icmp eq i32 %rem1, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %idxprom = sext i32 %0 to i64
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom
  %1 = load i32, i32 addrspace(1)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %1, i32 addrspace(1)* %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

; This test checks if the "packetize uniform" Vecz choice works on uniform
; values used by varying values, but not on uniform values used by other uniform
; values only.

; CHECK: define spir_kernel void @__vecz_v4_noreduce(ptr addrspace(3) %in, ptr addrspace(3) %out)
; CHECK: icmp ugt i64
; CHECK: and i32{{.*}}, 3
; CHECK: icmp eq i32
; CHECK: shl i32
; CHECK: ret void
