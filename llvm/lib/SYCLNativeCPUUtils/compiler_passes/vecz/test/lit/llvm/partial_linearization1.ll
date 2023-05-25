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

; RUN: veczc -k partial_linearization1 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;       a
;       |
;       b <-.
;      / \  |
;     c   d |
;    / \ /  |
;   e   f --'
;    \  |
;     \ g
;      \|
;       h
;
; * where nodes c and f are uniform branches, and node b is a varying
;   branch.
; * where nodes c, d, e, f, g and h are divergent.
;
; With partial linearization, it can be transformed in the following way:
;
;   a
;   |
;   b <.
;   |  |
;   d  |
;   |  |
;   c  |
;   |  |
;   f -'
;   |
;   g
;   |
;   e
;   |
;   h
;
; __kernel void partial_linearization1(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;   int i = 0;
;
;   while (1) {
;     if (id + i % 2 == 0) {
;       if (n > 2) {
;         goto e;
;       }
;     } else {
;       for (int i = 0; i < n + 10; i++) ret++;
;     }
;     if (n <= 2) break;
;   }
;
;   ret += n * 2;
;   for (int i = 0; i < n * 2; i++) ret -= i;
;   ret /= n;
;   goto early;
;
; e:
;   for (int i = 0; i < n + 5; i++) ret /= 2;
;   ret -= n;
;
; early:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization1(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end14, %entry
  %ret.0 = phi i32 [ 0, %entry ], [ %ret.2, %if.end14 ]
  %cmp = icmp eq i32 %conv, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  %cmp2 = icmp sgt i32 %n, 2
  br i1 %cmp2, label %e, label %if.end10

if.else:                                          ; preds = %while.body
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.else
  %ret.1 = phi i32 [ %ret.0, %if.else ], [ %inc, %for.body ]
  %storemerge = phi i32 [ 0, %if.else ], [ %inc9, %for.body ]
  %add6 = add nsw i32 %n, 10
  %cmp7 = icmp slt i32 %storemerge, %add6
  br i1 %cmp7, label %for.body, label %if.end10

for.body:                                         ; preds = %for.cond
  %inc = add nsw i32 %ret.1, 1
  %inc9 = add nsw i32 %storemerge, 1
  br label %for.cond

if.end10:                                         ; preds = %for.cond, %if.then
  %ret.2 = phi i32 [ %ret.0, %if.then ], [ %ret.1, %for.cond ]
  %cmp11 = icmp slt i32 %n, 3
  br i1 %cmp11, label %while.end, label %if.end14

if.end14:                                         ; preds = %if.end10
  br label %while.body

while.end:                                        ; preds = %if.end10
  %mul = mul i32 %n, 2
  %add15 = add nsw i32 %ret.2, %mul
  br label %for.cond17

for.cond17:                                       ; preds = %for.body21, %while.end
  %ret.3 = phi i32 [ %add15, %while.end ], [ %sub, %for.body21 ]
  %storemerge1 = phi i32 [ 0, %while.end ], [ %inc23, %for.body21 ]
  %mul18 = shl nsw i32 %n, 1
  %cmp19 = icmp slt i32 %storemerge1, %mul18
  br i1 %cmp19, label %for.body21, label %for.end24

for.body21:                                       ; preds = %for.cond17
  %sub = sub nsw i32 %ret.3, %storemerge1
  %inc23 = add nsw i32 %storemerge1, 1
  br label %for.cond17

for.end24:                                        ; preds = %for.cond17
  %0 = icmp eq i32 %ret.3, -2147483648
  %1 = icmp eq i32 %n, -1
  %2 = and i1 %1, %0
  %3 = icmp eq i32 %n, 0
  %4 = or i1 %3, %2
  %5 = select i1 %4, i32 1, i32 %n
  %div = sdiv i32 %ret.3, %5
  br label %early

e:                                                ; preds = %if.then
  br label %for.cond26

for.cond26:                                       ; preds = %for.body30, %e
  %ret.4 = phi i32 [ %ret.0, %e ], [ %div31, %for.body30 ]
  %storemerge3 = phi i32 [ 0, %e ], [ %inc33, %for.body30 ]
  %add27 = add nsw i32 %n, 5
  %cmp28 = icmp slt i32 %storemerge3, %add27
  br i1 %cmp28, label %for.body30, label %for.end34

for.body30:                                       ; preds = %for.cond26
  %div31 = sdiv i32 %ret.4, 2
  %inc33 = add nsw i32 %storemerge3, 1
  br label %for.cond26

for.end34:                                        ; preds = %for.cond26
  %sub35 = sub nsw i32 %ret.4, %n
  br label %early

early:                                            ; preds = %for.end34, %for.end24
  %storemerge2 = phi i32 [ %div, %for.end24 ], [ %sub35, %for.end34 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %storemerge2, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.kernels = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization1, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization1
; CHECK: br label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: br label %[[FORCONDPREHEADER:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[IFTHEN:.+]]:
; CHECK: br label %[[IFEND10:.+]]

; CHECK: [[FORCOND26PREHEADER:.+]]:
; CHECK: br label %[[FORCOND26:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[IFEND10LOOPEXIT:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[IFEND10LOOPEXIT]]:
; CHECK: br label %[[IFTHEN]]

; CHECK: [[IFEND10]]:
; CHECK: %[[CMP11:.+]] = icmp
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[WHILEEND:.+]]

; CHECK: [[WHILEEND]]:
; CHECK: br label %[[FORCOND17:.+]]

; CHECK: [[WHILEENDELSE:.+]]:
; CHECK: br label %[[FORCOND26PREHEADER]]

; CHECK: [[FORCOND17]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY21:.+]], label %[[FOREND24:.+]]

; CHECK: [[FORBODY21]]:
; CHECK: br label %[[FORCOND17]]

; CHECK: [[FOREND24]]:
; CHECK: br label %[[WHILEENDELSE]]

; CHECK: [[FORCOND26]]:
; CHECK: %[[CMP28:.+]] = icmp
; CHECK: br i1 %[[CMP28]], label %[[FORBODY30:.+]], label %[[FOREND34:.+]]

; CHECK: [[FORBODY30]]:
; CHECK: br label %[[FORCOND26]]

; CHECK: [[FOREND34]]:
; CHECK: br label %[[EARLY:.+]]

; CHECK: [[EARLY]]:
; CHECK: ret void
