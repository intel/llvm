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

; RUN: veczc -k varying_load1 -vecz-passes=cfg-convert -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
source_filename = "kernel.opencl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @varying_load1(i32 addrspace(1)* %out, i32 %n, i32 addrspace(1)* %meta) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %conv = trunc i64 %call to i32
  %cmp = icmp slt i32 %conv, 11
  br i1 %cmp, label %if.then, label %if.end16

if.then:                                          ; preds = %entry
  %0 = load i32, i32 addrspace(1)* %meta, align 4
  %cmp2 = icmp eq i32 %0, 0
  br i1 %cmp2, label %if.then4, label %if.end

if.then4:                                         ; preds = %if.then
  %mul5 = mul nsw i32 %conv, %n
  %1 = icmp eq i32 %mul5, -2147483648
  %2 = icmp eq i32 %n, -1
  %3 = and i1 %2, %1
  %4 = icmp eq i32 %n, 0
  %5 = or i1 %4, %3
  %6 = select i1 %5, i32 1, i32 %n
  %div6 = sdiv i32 %mul5, %6
  %add = add nsw i32 %div6, %conv
  %shl7 = mul i32 %add, 8
  %add8 = add nsw i32 %shl7, %mul5
  %shl9 = shl i32 %add8, 3
  br label %if.end

if.end:                                           ; preds = %if.then4, %if.then
  %sum.0 = phi i32 [ %shl9, %if.then4 ], [ %n, %if.then ]
  %rem1 = and i32 %conv, 1
  %cmp10 = icmp eq i32 %rem1, 0
  br i1 %cmp10, label %if.then12, label %if.end16

if.then12:                                        ; preds = %if.end
  %7 = load i32, i32 addrspace(1)* %meta, align 4
  %add13 = add nsw i32 %7, %n
  %mul14 = mul nsw i32 %add13, %sum.0
  br label %if.end16

if.end16:                                         ; preds = %if.end, %if.then12, %entry
  %ret.1 = phi i32 [ 0, %entry ], [ %mul14, %if.then12 ], [ 0, %if.end ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.1, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: convergent nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nobuiltin nounwind readonly }

; The purpose of this test is to make sure that if a condition is a use of a
; uniform load that is control dependent of a varying path, then the load will
; be considered "mask varying" and so the condition is still uniform.

; CHECK: spir_kernel void @__vecz_v4_varying_load1
; CHECK: if.then:
; CHECK: %{{.+}} = call i32 @__vecz_b_masked_load4
; CHECK: br i1
