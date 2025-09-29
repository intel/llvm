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

; RUN: veczc -k varying_load2 -vecz-passes=cfg-convert -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
source_filename = "kernel.opencl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @varying_load2(i32 addrspace(1)* %input, i32 addrspace(1)* %out) #0 {
entry:
  %call1 = call i64 @__mux_get_local_size(i32 0) #3
  %call2 = call i64 @__mux_get_local_id(i32 0) #3
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %input, i64 %call2
  %cmp = icmp ne i64 %call2, 0
  br i1 %cmp, label %for.cond.preheader, label %if.end14

for.cond.preheader:                               ; preds = %entry
  br label %for.cond

for.cond:                                         ; preds = %for.cond.preheader, %for.inc
  %max.0 = phi i32 [ %max.1, %for.inc ], [ 0, %for.cond.preheader ]
  %storemerge = phi i64 [ %inc, %for.inc ], [ 0, %for.cond.preheader ]
  %call6 = call i64 @__mux_get_local_size(i32 0) #3
  %cmp7 = icmp ult i64 %storemerge, %call6
  br i1 %cmp7, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %load1 = load i32, i32 addrspace(1)* %input, align 4
  %cmp9 = icmp ugt i32 %load1, %max.0
  br i1 %cmp9, label %if.then, label %for.inc

if.then:                                        ; preds = %for.body
  %load2 = load i32, i32 addrspace(1)* %input, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %max.1 = phi i32 [ %load2, %if.then ], [ %max.0, %for.body ]
  %inc = add i64 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %max.0.lcssa = phi i32 [ %max.0, %for.cond ]
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call1
  store i32 %max.0.lcssa, i32 addrspace(1)* %arrayidx13, align 4
  br label %if.end14

if.end14:                                         ; preds = %for.end, %entry
  ret void
}

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32) #1
; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_local_id(i32) #1
; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_local_size(i32) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent noduplicate "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nobuiltin nounwind readonly }
attributes #4 = { nounwind }

; The purpose of this test is to make sure that if a condition is a use of a
; uniform load that is control dependent of a varying path, then the load will
; be considered "mask varying" and so the condition is still uniform.

; CHECK: spir_kernel void @__vecz_v4_varying_load2
; CHECK: for.body:
; CHECK: %{{.+}} = call i32 @__vecz_b_masked_load4
; CHECK: br i1
; CHECK: if.then:
; CHECK: ret void
