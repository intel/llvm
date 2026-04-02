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

; RUN: veczc -k partial_linearization15 -vecz-passes="function(instcombine,simplifycfg),mergereturn,vecz-loop-rotate,function(loop(indvars)),cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;           a
;           |
;           b <-----.
;          / \      |
;         c   d     |
;        / \ /      |
;       /   e       |
;      /    |       |
;     /     g <---. |
;    /     / \    | |
;   f     h   i   | |
;   |    / \ / \  | |
;   |   |   j   k | |
;   |    \ / \ /  | |
;   |     l   m --' |
;   |    /          |
;   |   o ----------'
;   |   |
;   n   p
;    \ /
;     q
;
; * where nodes b, c, g, h, j and o are uniform branches, and node i is a
;   varying branch.
; * where nodes j, k, m, l, and o are divergent.
;
; With partial linearization, it will be transformed as follows:
;
;       a
;       |
;       b <-----.
;      / \      |
;     c   d     |
;    / \ /      |
;   f   e       |
;   |   |       |
;   |   g <---. |
;   |  / \    | |
;   | h   i   | |
;   | |   |   | |
;   | |   k   | |
;   |  \ /    | |
;   |   j     | |
;   |   |     | |
;   |   m ----' |
;   |   |       |
;   |   l       |
;   |   |       |
;   |   o ------'
;   |   |
;   n   p
;    \ /
;     q
;
; __kernel void partial_linearization15(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   while (1) {
;     if (n > 0) { // b
;       // c
;       for (int i = 0; i < n * 2; i++) ret++;
;       if (n <= 10) {
;         // f
;         goto f;
;       }
;     } else {
;       // d
;       for (int i = 0; i < n / 4; i++) ret++;
;     }
;     // e
;     ret++;
;     while (1) {
;       if (n & 1) { // g
;         // h
;         if (n < 3) {
;           goto l;
;         }
;       } else {
;         // i
;         if (ret + id >= n) {
;           // k
;           ret /= n * n + ret;
;           goto m;
;         }
;       }
;       // j
;       if (n & 1) {
;         goto l;
;       }
;       // m
; m:
;       ret++;
;     }
; l:
;     ret *= 4;
; o:
;     if (n & 1) {
;       // p
;       ret++;
;       goto p;
;     }
;   }
;
; p:
;   for (int i = 0; i < n / 4; i++) ret++;
;   goto q;
;
; f:
;   ret /= n;
;   goto n;
;
; n:
;   for (int i = 0; i < n * 2; i++) ret++;
;
; q:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization15(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %l, %entry
  %ret.0 = phi i32 [ 0, %entry ], [ %mul40, %l ]
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.cond, label %for.cond9

for.cond:                                         ; preds = %for.body, %while.body
  %ret.1 = phi i32 [ %inc, %for.body ], [ %ret.0, %while.body ]
  %storemerge3 = phi i32 [ %inc4, %for.body ], [ 0, %while.body ]
  %mul = shl nsw i32 %n, 1
  %cmp2 = icmp slt i32 %storemerge3, %mul
  br i1 %cmp2, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %inc = add nsw i32 %ret.1, 1
  %inc4 = add nuw nsw i32 %storemerge3, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %cmp5 = icmp slt i32 %n, 11
  br i1 %cmp5, label %f, label %if.end17

for.cond9:                                        ; preds = %for.body12, %while.body
  %ret.2 = phi i32 [ %inc13, %for.body12 ], [ %ret.0, %while.body ]
  %storemerge = phi i32 [ %inc15, %for.body12 ], [ 0, %while.body ]
  %div = sdiv i32 %n, 4
  %cmp10 = icmp slt i32 %storemerge, %div
  br i1 %cmp10, label %for.body12, label %if.end17

for.body12:                                       ; preds = %for.cond9
  %inc13 = add nsw i32 %ret.2, 1
  %inc15 = add nuw nsw i32 %storemerge, 1
  br label %for.cond9

if.end17:                                         ; preds = %for.cond9, %for.end
  %ret.3 = phi i32 [ %ret.1, %for.end ], [ %ret.2, %for.cond9 ]
  br label %while.body20

while.body20:                                     ; preds = %m, %if.end17
  %storemerge1.in = phi i32 [ %ret.3, %if.end17 ], [ %ret.4, %m ]
  %storemerge1 = add nsw i32 %storemerge1.in, 1
  %and = and i32 %n, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.else26, label %if.then21

if.then21:                                        ; preds = %while.body20
  %cmp22 = icmp slt i32 %n, 3
  br i1 %cmp22, label %l, label %if.end34

if.else26:                                        ; preds = %while.body20
  %add = add nsw i32 %storemerge1, %conv
  %cmp27 = icmp slt i32 %add, %n
  br i1 %cmp27, label %if.end34, label %if.then29

if.then29:                                        ; preds = %if.else26
  %mul30 = mul nsw i32 %n, %n
  %add31 = add nsw i32 %storemerge1, %mul30
  %0 = icmp eq i32 %add31, 0
  %1 = select i1 %0, i32 1, i32 %add31
  %div32 = sdiv i32 %storemerge1, %1
  br label %m

if.end34:                                         ; preds = %if.else26, %if.then21
  %and35 = and i32 %n, 1
  %tobool36 = icmp eq i32 %and35, 0
  br i1 %tobool36, label %m, label %l

m:                                                ; preds = %if.end34, %if.then29
  %ret.4 = phi i32 [ %div32, %if.then29 ], [ %storemerge1, %if.end34 ]
  br label %while.body20

l:                                                ; preds = %if.end34, %if.then21
  %mul40 = shl nsw i32 %storemerge1, 2
  %and41 = and i32 %n, 1
  %tobool42 = icmp eq i32 %and41, 0
  br i1 %tobool42, label %while.body, label %if.then43

if.then43:                                        ; preds = %l
  %inc44 = or i32 %mul40, 1
  br label %for.cond47

for.cond47:                                       ; preds = %for.body51, %if.then43
  %ret.5 = phi i32 [ %inc44, %if.then43 ], [ %inc52, %for.body51 ]
  %storemerge2 = phi i32 [ 0, %if.then43 ], [ %inc54, %for.body51 ]
  %div48 = sdiv i32 %n, 4
  %cmp49 = icmp slt i32 %storemerge2, %div48
  br i1 %cmp49, label %for.body51, label %q

for.body51:                                       ; preds = %for.cond47
  %inc52 = add nsw i32 %ret.5, 1
  %inc54 = add nuw nsw i32 %storemerge2, 1
  br label %for.cond47

f:                                                ; preds = %for.end
  %2 = icmp eq i32 %ret.1, -2147483648
  %3 = icmp eq i32 %n, -1
  %4 = and i1 %3, %2
  %5 = icmp eq i32 %n, 0
  %6 = or i1 %5, %4
  %7 = select i1 %6, i32 1, i32 %n
  %div56 = sdiv i32 %ret.1, %7
  br label %for.cond59

for.cond59:                                       ; preds = %for.body63, %f
  %ret.6 = phi i32 [ %div56, %f ], [ %inc64, %for.body63 ]
  %storemerge4 = phi i32 [ 0, %f ], [ %inc66, %for.body63 ]
  %mul60 = shl nsw i32 %n, 1
  %cmp61 = icmp slt i32 %storemerge4, %mul60
  br i1 %cmp61, label %for.body63, label %q

for.body63:                                       ; preds = %for.cond59
  %inc64 = add nsw i32 %ret.6, 1
  %inc66 = add nuw nsw i32 %storemerge4, 1
  br label %for.cond59

q:                                                ; preds = %for.cond59, %for.cond47
  %ret.7 = phi i32 [ %ret.5, %for.cond47 ], [ %ret.6, %for.cond59 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.7, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.kernels = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization15, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization15
; CHECK: br label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[FORCONDPREHEADER:.+]], label %[[FORCOND9PREHEADER:.+]]

; CHECK: [[FORCOND9PREHEADER]]:
; CHECK: br label %[[FORCOND9:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 false, label %[[FORBODY:.+]], label %[[FOREND:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[FOREND]]:
; CHECK: %[[CMP5:.+]] = icmp
; CHECK: br i1 %[[CMP5]], label %[[F:.+]], label %[[IFEND17:.+]]

; CHECK: [[FORCOND9]]:
; CHECK: %[[CMP10:.+]] = icmp
; CHECK: br i1 %[[CMP10]], label %[[FORBODY12:.+]], label %[[IFEND17LOOPEXIT:.+]]

; CHECK: [[FORBODY12]]:
; CHECK: br label %[[FORCOND9]]

; CHECK: [[IFEND17LOOPEXIT]]:
; CHECK: br label %[[IFEND17]]

; CHECK: [[IFEND17]]:
; CHECK: br label %[[WHILEBODY20:.+]]

; CHECK: [[WHILEBODY20]]:
; CHECK: %[[AND:.+]] = and i32
; CHECK: %[[TOBOOL:.+]] = icmp eq i32 %[[AND]]
; CHECK: br i1 %[[TOBOOL]], label %[[IFELSE26:.+]], label %[[IFTHEN21:.+]]

; CHECK: [[IFTHEN21]]:
; CHECK: br label %[[M:.+]]

; CHECK: [[IFELSE26]]:
; CHECK: br label %[[IFTHEN29:.+]]

; CHECK: [[IFTHEN29]]:
; CHECK: br label %[[IFEND34:.+]]

; CHECK: [[IFEND34]]:
; CHECK: br label %[[M:.+]]

; CHECK: [[M]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY20]], label %[[WHILEBODY20PUREEXIT:.+]]

; CHECK: [[WHILEBODY20PUREEXIT]]:
; CHECK: br label %[[L:.+]]

; CHECK: [[L]]:
; CHECK: %[[TOBOOL42:.+]] = icmp
; CHECK: br i1 %[[TOBOOL42]], label %[[WHILEBODY]], label %[[IFTHEN43:.+]]

; CHECK: [[IFTHEN43]]:
; CHECK: br label %[[FORCOND47:.+]]

; CHECK: [[FORCOND47]]:
; CHECK: %[[CMP49:.+]] = icmp
; CHECK: br i1 %[[CMP49]], label %[[FORBODY51:.+]], label %[[QLOOPEXIT2:.+]]

; CHECK: [[FORBODY51]]:
; CHECK: br label %[[FORCOND47]]

; CHECK: [[F]]:
; CHECK: br label %[[FORCOND59:.+]]

; CHECK: [[FORCOND59]]:
; CHECK: br i1 false, label %[[FORBODY63:.+]], label %[[QLOOPEXIT:.+]]

; CHECK: [[FORBODY63]]:
; CHECK: br label %[[FORCOND59]]

; CHECK: [[QLOOPEXIT]]:
; CHECK: br label %[[Q:.+]]

; CHECK: [[QLOOPEXIT2]]:
; CHECK: br label %[[Q]]

; CHECK: [[Q]]:
; CHECK: ret void
