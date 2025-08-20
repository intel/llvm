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

; RUN: veczc -k vector_loop -vecz-simd-width=4 -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @vector_loop(i32 addrspace(1)* %in, <4 x i32> addrspace(1)* %in2, i32 addrspace(1)* %out) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %initaddr = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %in2, i64 %call
  %init = load <4 x i32>, <4 x i32> addrspace(1)* %initaddr
  %cmp = icmp eq i64 %call, 0
  br i1 %cmp, label %for.end, label %for.cond

for.cond:                                         ; preds = %entry, %for.body
  %storemerge = phi <4 x i32> [ %inc, %for.body ], [ %init, %entry ]
  %call1 = call i64 @__mux_get_global_size(i32 0)
  %conv = trunc i64 %call1 to i32
  %0 = extractelement <4 x i32> %storemerge, i64 0
  %cmp2 = icmp slt i32 %0, %conv
  br i1 %cmp2, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = extractelement <4 x i32> %storemerge, i64 0
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom
  %2 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %3 = extractelement <4 x i32> %storemerge, i64 0
  %idxprom3 = sext i32 %3 to i64
  %arrayidx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom3
  store i32 %2, i32 addrspace(1)* %arrayidx4, align 4
  %4 = extractelement <4 x i32> %storemerge, i64 1
  %idxprom5 = sext i32 %4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom5
  %5 = load i32, i32 addrspace(1)* %arrayidx6, align 4
  %6 = extractelement <4 x i32> %storemerge, i64 1
  %idxprom7 = sext i32 %6 to i64
  %arrayidx8 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom7
  store i32 %5, i32 addrspace(1)* %arrayidx8, align 4
  %7 = extractelement <4 x i32> %storemerge, i64 2
  %idxprom9 = sext i32 %7 to i64
  %arrayidx10 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom9
  %8 = load i32, i32 addrspace(1)* %arrayidx10, align 4
  %9 = extractelement <4 x i32> %storemerge, i64 2
  %idxprom11 = sext i32 %9 to i64
  %arrayidx12 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom11
  store i32 %8, i32 addrspace(1)* %arrayidx12, align 4
  %10 = extractelement <4 x i32> %storemerge, i64 3
  %idxprom13 = sext i32 %10 to i64
  %arrayidx14 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom13
  %11 = load i32, i32 addrspace(1)* %arrayidx14, align 4
  %12 = extractelement <4 x i32> %storemerge, i64 3
  %idxprom15 = sext i32 %12 to i64
  %arrayidx16 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom15
  store i32 %11, i32 addrspace(1)* %arrayidx16, align 4
  %inc = add <4 x i32> %storemerge, <i32 1, i32 1, i32 1, i32 1>
  br label %for.cond

for.end:                                          ; preds = %entry, %for.cond
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare i64 @__mux_get_global_size(i32)

; This test checks if a varying <4 x i32> phi gets scalarized
; if it is only accessed through individually extracted elements.
; CHECK: define spir_kernel void @__vecz_v4_vector_loop
; CHECK: %storemerge{{[0-9]+}} = phi <4 x i32> [ %{{[0-9]+}}, %entry.ROSCC ], [ %inc{{[0-9]+}}, %for.cond ]
; CHECK: %storemerge{{[0-9]+}} = phi <4 x i32> [ %{{[0-9]+}}, %entry.ROSCC ], [ %inc{{[0-9]+}}, %for.cond ]
; CHECK: %storemerge{{[0-9]+}} = phi <4 x i32> [ %{{[0-9]+}}, %entry.ROSCC ], [ %inc{{[0-9]+}}, %for.cond ]
; CHECK: %storemerge{{[0-9]+}} = phi <4 x i32> [ %{{[0-9]+}}, %entry.ROSCC ], [ %inc{{[0-9]+}}, %for.cond ]
; CHECK: %inc{{[0-9]+}} = add <4 x i32> %storemerge{{[0-9]+}}, {{<(i32 1(, )?)+>|splat \(i32 1\)}}
; CHECK: %inc{{[0-9]+}} = add <4 x i32> %storemerge{{[0-9]+}}, {{<(i32 1(, )?)+>|splat \(i32 1\)}}
; CHECK: %inc{{[0-9]+}} = add <4 x i32> %storemerge{{[0-9]+}}, {{<(i32 1(, )?)+>|splat \(i32 1\)}}
; CHECK: %inc{{[0-9]+}} = add <4 x i32> %storemerge{{[0-9]+}}, {{<(i32 1(, )?)+>|splat \(i32 1\)}}
; CHECK: ret void
