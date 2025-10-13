; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: veczc -k reduce -vecz-choices=PacketizeUniform -vecz-simd-width=4 -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_local_id(i32)
declare i64 @__mux_get_global_id(i32)
declare i64 @__mux_get_local_size(i32)

; Function Attrs: nounwind
define spir_kernel void @reduce(i32 addrspace(3)* %in, i32 addrspace(3)* %out) {
entry:
  %call = call i64 @__mux_get_local_id(i32 0)
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %storemerge = phi i32 [ 1, %entry ], [ %mul6, %for.inc ]
  %conv = zext i32 %storemerge to i64
  %call1 = call i64 @__mux_get_local_size(i32 0)
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

; This test checks if the "packetize uniform" Vecz choice works on uniform
; values used by varying values, but not on uniform values used by other uniform
; values only.

; CHECK: define spir_kernel void @__vecz_v4_reduce(ptr addrspace(3) %in, ptr addrspace(3) %out)
; CHECK: insertelement <4 x i64> poison, i64 %{{.+}}, {{(i32|i64)}} 0
; CHECK: shufflevector <4 x i64> %{{.+}}, <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK: phi <4 x i32>
; CHECK: mul <4 x i32> %{{.+}}, {{<(i32 3(, )?)+>|splat \(i32 3\)}}
; CHECK: urem <4 x i64>
; CHECK: icmp eq <4 x i64> %{{.+}}, zeroinitializer

; The branch condition is actually Uniform, despite the divergence analysis
; CHECK: icmp ugt i64
; CHECK: ret void
