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

; RUN: veczc -vecz-fail-quietly -k test -vecz-passes="cfg-convert" -S < %s

; This tests only that the kernel does not crash the vectorizer.

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @test(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %e, %entry
  %n.off = add i32 %n, -1
  %0 = icmp ult i32 %n.off, 4
  %cmp6 = icmp slt i32 %n, 3
  %or.cond1 = or i1 %cmp6, %0
  br i1 %or.cond1, label %f, label %if.else

while.body5:                                      ; preds = %d
  switch i32 %n, label %g [
    i32 3, label %if.else
    i32 2, label %h
  ]

if.else:                                          ; preds = %while.body5, %while.body
  %cmp9 = icmp sge i32 %conv, %n
  %and = and i32 %n, 1
  %tobool = icmp eq i32 %and, 0
  %or.cond2 = or i1 %tobool, %cmp9
  br i1 %or.cond2, label %d, label %h

d:                                                ; preds = %if.else
  %cmp16 = icmp sgt i32 %n, 3
  br i1 %cmp16, label %e, label %while.body5

e:                                                ; preds = %d
  %and20 = and i32 %n, 1
  %tobool21 = icmp eq i32 %and20, 0
  br i1 %tobool21, label %while.body, label %g

f:                                                ; preds = %while.body
  %cmp24 = icmp eq i32 %n, 2
  br i1 %cmp24, label %h, label %g

g:                                                ; preds = %f, %e, %while.body5
  br label %for.cond

for.cond:                                         ; preds = %for.body, %g
  %ret.0 = phi i32 [ 0, %g ], [ %inc, %for.body ]
  %storemerge = phi i32 [ 0, %g ], [ %inc31, %for.body ]
  %cmp29 = icmp sgt i32 %storemerge, %n
  br i1 %cmp29, label %h, label %for.body

for.body:                                         ; preds = %for.cond
  %inc = add nuw nsw i32 %ret.0, 1
  %inc31 = add nuw nsw i32 %storemerge, 1
  br label %for.cond

h:                                                ; preds = %for.cond, %f, %if.else, %while.body5
  %ret.1 = phi i32 [ 0, %f ], [ %ret.0, %for.cond ], [ 0, %if.else ], [ 0, %while.body5 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.1, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nobuiltin nounwind readonly }
