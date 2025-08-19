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

; RUN: veczc -k phi_memory -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @phi_memory(i32 addrspace(1)* %input, i32 addrspace(1)* %output, i32 %size) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  %idx.ext = sext i32 %conv to i64
  %add.ptr = getelementptr inbounds i32, i32 addrspace(1)* %output, i64 %idx.ext
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %output.addr.0 = phi i32 addrspace(1)* [ %add.ptr, %entry ], [ %add.ptr2, %for.body ]
  %storemerge = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %storemerge, %size
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %storemerge, %conv
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %input, i64 %idxprom
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  store i32 %0, i32 addrspace(1)* %output.addr.0, align 4
  %add.ptr2 = getelementptr inbounds i32, i32 addrspace(1)* %output.addr.0, i64 1
  %inc = add nsw i32 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare i64 @__mux_get_global_id(i32) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nobuiltin nounwind }

; It checks that the contiguity of the load and store is identified through the
; loop-incrementing pointer PHI node
;
; CHECK: void @__vecz_v4_phi_memory
; CHECK: %[[LD:.+]] = load <4 x i32>
; CHECK: store <4 x i32> %[[LD]]
; CHECK-NOT: scatter_store
; CHECK-NOT: gather_load
