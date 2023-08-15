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

; RUN: veczc -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Check if the call to max in the if block has been replaced with its vector
; equivalent
; CHECK: call spir_func <[[WIDTH:[0-9]+]] x i32> @_Z3maxDv[[WIDTH]]_iS_(<[[WIDTH]] x i32> {{.+}}, <[[WIDTH]] x i32> {{.+}})
; CHECK: call spir_func <[[WIDTH]] x i32> @_Z3maxDv[[WIDTH]]_iS_(<[[WIDTH]] x i32> {{.+}}, <[[WIDTH]] x i32> {{.+}})

; There shouldn't be any masked versions of max
; CHECK-NOT: masked_Z3max

define spir_kernel void @entry(ptr addrspace(1) %input, ptr addrspace(1) %output) {
entry:
  %call = tail call i64 @__mux_get_local_id(i32 0)
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %call
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %call
  %1 = load i32, ptr addrspace(1) %arrayidx2, align 4
  %add = add nsw i32 %0, 1
  %add3 = add nsw i32 %1, 1
  %call4 = tail call spir_func i32 @_Z3maxii(i32 %add, i32 %add3)
  %add.i = shl nsw i32 %call4, 1
  %idxprom.i = sext i32 %add.i to i64
  %arrayidx.i = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %idxprom.i
  store i32 %add.i, ptr addrspace(1) %arrayidx.i, align 4
  %2 = load i32, ptr addrspace(1) %arrayidx2, align 4
  %3 = load i32, ptr addrspace(1) %arrayidx, align 4
  %4 = icmp eq i32 %2, -2147483648
  %5 = icmp eq i32 %3, -1
  %6 = and i1 %4, %5
  %7 = icmp eq i32 %3, 0
  %8 = or i1 %7, %6
  %9 = select i1 %8, i32 1, i32 %3
  %10 = icmp eq i32 %9, -1
  %11 = and i1 %4, %10
  %12 = select i1 %11, i32 1, i32 %9
  %rem = srem i32 %2, %12
  %tobool.not = icmp eq i32 %rem, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %call9 = tail call spir_func i32 @_Z3maxii(i32 %0, i32 %1)
  %add.i27 = shl nsw i32 %call9, 1
  %idxprom.i28 = sext i32 %add.i27 to i64
  %arrayidx.i29 = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %idxprom.i28
  store i32 %add.i27, ptr addrspace(1) %arrayidx.i29, align 4
  br label %if.end

if.end:
  %idxprom.i31.pre-phi = phi i64 [ %idxprom.i28, %if.then ], [ %idxprom.i, %entry ]
  %add.i30.pre-phi = phi i32 [ %add.i27, %if.then ], [ %add.i, %entry ]
  %r.0 = phi i32 [ %call9, %if.then ], [ %call4, %entry ]
  %arrayidx.i32 = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %idxprom.i31.pre-phi
  store i32 %add.i30.pre-phi, ptr addrspace(1) %arrayidx.i32, align 4
  store i32 %r.0, ptr addrspace(1) %arrayidx2, align 4
  ret void
}

declare i64 @__mux_get_local_id(i32)

declare spir_func i32 @_Z3maxii(i32, i32)

declare spir_func <4 x i32> @_Z3maxDv4_iS_(<4 x i32>, <4 x i32>)
