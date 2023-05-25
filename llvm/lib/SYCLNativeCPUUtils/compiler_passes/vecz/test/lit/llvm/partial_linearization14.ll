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

; RUN: veczc -k partial_linearization14 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;     a
;    / \
;   b   c <-.
;   |  / \  |
;   | d   e |
;   |/ \ /  |
;   f   g --'
;    \  |
;     \ h
;      \|
;       i
;
; * where nodes a, d and g are uniform branches, and node c is a varying
;   branch.
; * where nodes d, e, f, g, h and i are divergent.
;
; With partial linearization, it can be transformed in the following way:
;
;     a
;    / \
;   b   c <.
;   |   |  |
;   |   e  |
;   |   |  |
;   |   d  |
;   |   |  |
;   |   g -'
;    \  |
;     \ h
;      \|
;       f
;       |
;       i
;
; __kernel void partial_linearization14(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;   int i = 0;
;
;   if (n < 5) {
;     for (int i = 0; i < n + 10; i++) ret++;
;     goto f;
;   } else {
;     while (1) {
;       if (id + i % 2 == 0) {
;         if (n > 2) {
;           goto f;
;         }
;       } else {
;         for (int i = 0; i < n + 10; i++) ret++;
;       }
;       if (n <= 2) break;
;     }
;   }
;
;   ret += n * 2;
;   for (int i = 0; i < n * 2; i++) ret -= i;
;   ret /= n;
;   goto early;
;
; f:
;   for (int i = 0; i < n + 5; i++) ret /= 2;
;   ret -= n;
;
; early:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization14(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %conv = trunc i64 %call to i32
  %cmp = icmp slt i32 %n, 5
  br i1 %cmp, label %for.cond, label %while.body

for.cond:                                         ; preds = %for.body, %entry
  %ret.0 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %storemerge4 = phi i32 [ %inc5, %for.body ], [ 0, %entry ]
  %add = add nsw i32 %n, 10
  %cmp3 = icmp slt i32 %storemerge4, %add
  br i1 %cmp3, label %for.body, label %f

for.body:                                         ; preds = %for.cond
  %inc = add nuw nsw i32 %ret.0, 1
  %inc5 = add nuw nsw i32 %storemerge4, 1
  br label %for.cond

while.body:                                       ; preds = %if.end24, %entry
  %ret.1 = phi i32 [ 0, %entry ], [ %ret.3, %if.end24 ]
  %cmp7 = icmp eq i32 %conv, 0
  br i1 %cmp7, label %if.then9, label %for.cond15

if.then9:                                         ; preds = %while.body
  %cmp10 = icmp sgt i32 %n, 2
  br i1 %cmp10, label %f, label %if.end24

for.cond15:                                       ; preds = %for.body19, %while.body
  %ret.2 = phi i32 [ %inc20, %for.body19 ], [ %ret.1, %while.body ]
  %storemerge = phi i32 [ %inc22, %for.body19 ], [ 0, %while.body ]
  %add16 = add nsw i32 %n, 10
  %cmp17 = icmp slt i32 %storemerge, %add16
  br i1 %cmp17, label %for.body19, label %if.end24

for.body19:                                       ; preds = %for.cond15
  %inc20 = add nsw i32 %ret.2, 1
  %inc22 = add nuw nsw i32 %storemerge, 1
  br label %for.cond15

if.end24:                                         ; preds = %for.cond15, %if.then9
  %ret.3 = phi i32 [ %ret.1, %if.then9 ], [ %ret.2, %for.cond15 ]
  %cmp25 = icmp slt i32 %n, 3
  br i1 %cmp25, label %if.end29, label %while.body

if.end29:                                         ; preds = %if.end24
  %mul = mul i32 %n, 2
  %add30 = add nsw i32 %ret.3, %mul
  br label %for.cond32

for.cond32:                                       ; preds = %for.body36, %if.end29
  %ret.4 = phi i32 [ %add30, %if.end29 ], [ %sub, %for.body36 ]
  %storemerge1 = phi i32 [ 0, %if.end29 ], [ %inc38, %for.body36 ]
  %mul33 = shl nsw i32 %n, 1
  %cmp34 = icmp slt i32 %storemerge1, %mul33
  br i1 %cmp34, label %for.body36, label %for.end39

for.body36:                                       ; preds = %for.cond32
  %sub = sub nsw i32 %ret.4, %storemerge1
  %inc38 = add nuw nsw i32 %storemerge1, 1
  br label %for.cond32

for.end39:                                        ; preds = %for.cond32
  %0 = icmp eq i32 %ret.4, -2147483648
  %1 = icmp eq i32 %n, -1
  %2 = and i1 %1, %0
  %3 = icmp eq i32 %n, 0
  %4 = or i1 %3, %2
  %5 = select i1 %4, i32 1, i32 %n
  %div = sdiv i32 %ret.4, %5
  br label %early

f:                                                ; preds = %if.then9, %for.cond
  %ret.5 = phi i32 [ %ret.0, %for.cond ], [ %ret.1, %if.then9 ]
  br label %for.cond41

for.cond41:                                       ; preds = %for.body45, %f
  %ret.6 = phi i32 [ %ret.5, %f ], [ %div46, %for.body45 ]
  %storemerge3 = phi i32 [ 0, %f ], [ %inc48, %for.body45 ]
  %add42 = add nsw i32 %n, 5
  %cmp43 = icmp slt i32 %storemerge3, %add42
  br i1 %cmp43, label %for.body45, label %for.end49

for.body45:                                       ; preds = %for.cond41
  %div46 = sdiv i32 %ret.6, 2
  %inc48 = add nuw nsw i32 %storemerge3, 1
  br label %for.cond41

for.end49:                                        ; preds = %for.cond41
  %sub50 = sub nsw i32 %ret.6, %n
  br label %early

early:                                            ; preds = %for.end49, %for.end39
  %storemerge2 = phi i32 [ %div, %for.end39 ], [ %sub50, %for.end49 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %storemerge2, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: convergent nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.kernels = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization14, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization14
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[FORCONDPREHEADER:.+]], label %[[WHILEBODYPREHEADER:.+]]

; CHECK: [[WHILEBODYPREHEADER]]:
; CHECK: br label %[[WHILEBODY:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[FLOOPEXIT:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[WHILEBODY]]:
; CHECK: br label %[[FORCOND15PREHEADER:.+]]

; CHECK: [[FORCOND15PREHEADER]]:
; CHECK: br label %[[FORCOND15:.+]]

; CHECK: [[IFTHEN9:.+]]:
; CHECK: br label %[[IFEND24:.+]]

; CHECK: [[FORCOND15]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY19:.+]], label %[[IFEND24LOOPEXIT:.+]]

; CHECK: [[FORBODY19]]:
; CHECK: br label %[[FORCOND15]]

; CHECK: [[IFEND24LOOPEXIT]]:
; CHECK: br label %[[IFTHEN9]]

; CHECK: [[IFEND24]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[IFEND29:.+]]

; CHECK: [[IFEND29]]:
; CHECK: br label %[[FORCOND32:.+]]

; CHECK: [[IFEND29ELSE:.+]]:
; CHECK: br label %[[FLOOPEXIT2:.+]]

; CHECK: [[FORCOND32]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY36:.+]], label %[[FOREND39:.+]]

; CHECK: [[FORBODY36]]:
; CHECK: br label %[[FORCOND32]]

; CHECK: [[FOREND39]]:
; CHECK: br label %[[IFEND29ELSE]]

; CHECK: [[FLOOPEXIT]]:
; CHECK: br label %[[F:.+]]

; CHECK: [[FLOOPEXIT2]]:
; CHECK: br label %[[F]]

; CHECK: [[F]]:
; CHECK: br label %[[FORCOND41:.+]]

; CHECK: [[FORCOND41]]:
; CHECK: %[[CMP43:.+]] = icmp
; CHECK: br i1 %[[CMP43]], label %[[FORBODY45:.+]], label %[[FOREND49:.+]]

; CHECK: [[FORBODY45]]:
; CHECK: br label %[[FORCOND41]]

; CHECK: [[FOREND49]]:
; CHECK: br label %[[EARLY:.+]]

; CHECK: [[EARLY]]:
; CHECK: ret void
