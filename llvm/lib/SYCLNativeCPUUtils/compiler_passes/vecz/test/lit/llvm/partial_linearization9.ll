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

; RUN: veczc -k partial_linearization9 -vecz-passes=cfg-convert,cleanup-divergence -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;   a
;   |
;   b <--.
;   |    |
;   c <. |
;   |  | |
;   d -' |
;   |    |
;   e ---'
;   |
;   f
;
; * where node e is a varying branch.
; * where node f is divergent.
;
; With partial linearization we will have a CFG of the form:
;
;   a
;   |
;   b <--.
;   |    |
;   c <. |
;   |  | |
;   d -' |
;   |    |
;   e ---'
;   |
;   f
;
; __kernel void partial_linearization9(__global int *out, int n) {
;   int id = get_global_id(0);
;   int i = 0;
;
;   while (1) {
;     int j = 0;
;     for (; ; i++) {
;       if (j++ > n) break;
;     }
;     if (i++ + id > n) break;
;   }
;
;   out[id] = i;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization9(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end7, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc3, %if.end7 ]
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %while.body
  %i.1 = phi i32 [ %i.0, %while.body ], [ %inc3, %for.inc ]
  %j.0 = phi i32 [ 0, %while.body ], [ %inc, %for.inc ]
  %cmp = icmp sgt i32 %j.0, %n
  %inc3 = add nsw i32 %i.1, 1
  br i1 %cmp, label %for.end, label %for.inc

for.inc:                                          ; preds = %for.cond
  %inc = add nsw i32 %j.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %add = add nsw i32 %i.1, %conv
  %cmp4 = icmp sgt i32 %add, %n
  br i1 %cmp4, label %while.end, label %if.end7

if.end7:                                          ; preds = %for.end
  br label %while.body

while.end:                                        ; preds = %for.end
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %inc3, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind readonly
declare i64 @__mux_get_global_id(i32) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.kernels = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization9, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization9
; CHECK: br label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(true)}}, label %[[FOREND:.+]], label %[[FORINC:.+]]

; CHECK: [[FORINC]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[FOREND]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[WHILEEND:.+]]

; CHECK: [[WHILEEND]]:
; CHECK: ret void
