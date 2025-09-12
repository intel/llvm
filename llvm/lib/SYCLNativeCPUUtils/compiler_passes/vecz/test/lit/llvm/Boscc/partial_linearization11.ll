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

; RUN: veczc -k partial_linearization11 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;       a
;       |
;       b <-------.
;       |         |
;       c <---.   |
;      / \    |   |
;     d   e   |   |
;    / \ / \  |   |
;   i   f   g |   |
;   |  / \ / \|   |
;   | j   h --'   |
;   | |        \  |
;   | |         k |
;   |  \       /  |
;   |   \     /   |
;   |    \   /    |
;   |     \ /     |
;   |      l -----'
;   |     /
;    \   m
;     \ /
;      n
;
; * where nodes c, d, f, g, and l are uniform branches, and node e is a
;   varying branch.
; * where nodes i, f, g, j, h, k, l, m and n are divergent.
;
; With BOSCC, it will be transformed as follows:
;
;       a
;       |
;       b <-------.    b' <----.
;       |         |    |       |
;       c <---.   |    c' <--. |
;      / \    |   |   / \    | |
;     d   e___|___|_ d'  e'  | |
;    / \ / \  |   | \|__ |   | |
;   i   f   g |   |  |  `g'  | |
;   |  / \ / \|   |   \ /    | |
;   | j   h --'   |    f'    | |
;   | |        \  |    |     | |
;   | |         k |    h' ---' |
;   |  \       /  |    |       |
;   |   \     /   |    k'      |
;   |    \   /    |    |       |
;   |     \ /     |    j'      |
;   |      l -----'    |       |
;   |     /            l' -----'
;    \   m             |
;     \ /              m'
;      n               |
;      |               i'
;      |               |
;      `-----> & <---- n'
;
; where '&' represents merge blocks of BOSCC regions.
;
; __kernel void partial_linearization11(__global int *out, int n) {
;   // a
;   int id = get_global_id(0);
;   int ret = 0;
;
;   while (1) {
;     // b
;     while (1) {
;       if (n < 5) { // c
;         // d
;         for (int i = 0; i < n * 2; i++) ret++;
;         if (n <= 3) {
;           // i
;           goto i;
;         }
;       } else {
;         // e
;         if (ret + id >= n) {
;           // g
;           ret /= n * n + ret;
;           if (n <= 10) {
;             goto k;
;           } else {
;             goto h;
;           }
;         }
;       }
;       // f
;       ret *= n;
;       if (n & 1) {
;         goto j;
;       }
;
;       // h
; h:
;       ret++;
;     }
;
; j:
;     ret += n * 2 + 20;
;     goto l;
;
; k:
;     ret *= n;
;     goto l;
;
; l:
;     if (n & 1) {
;       // m
;       ret++;
;       goto m;
;     }
;   }
;
; m:
;   for (int i = 0; i < n / 4; i++) ret++;
;   goto n;
;
; i:
;   ret /= n;
;
; n:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization11(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end33, %entry
  %ret.0 = phi i32 [ 0, %entry ], [ %storemerge, %if.end33 ]
  br label %while.body2

while.body2:                                      ; preds = %h, %while.body
  %ret.1 = phi i32 [ %ret.0, %while.body ], [ %inc24, %h ]
  %cmp = icmp slt i32 %n, 5
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %while.body2
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.then
  %ret.2 = phi i32 [ %ret.1, %if.then ], [ %inc, %for.body ]
  %storemerge2 = phi i32 [ 0, %if.then ], [ %inc6, %for.body ]
  %mul = shl nsw i32 %n, 1
  %cmp4 = icmp slt i32 %storemerge2, %mul
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %inc = add nsw i32 %ret.2, 1
  %inc6 = add nsw i32 %storemerge2, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %cmp7 = icmp slt i32 %n, 4
  br i1 %cmp7, label %i44, label %if.end20

if.else:                                          ; preds = %while.body2
  %add = add nsw i32 %ret.1, %conv
  %cmp10 = icmp slt i32 %add, %n
  br i1 %cmp10, label %if.end20, label %if.then12

if.then12:                                        ; preds = %if.else
  %mul13 = mul nsw i32 %n, %n
  %add14 = add nsw i32 %ret.1, %mul13
  %0 = icmp eq i32 %ret.1, -2147483648
  %1 = icmp eq i32 %add14, -1
  %2 = and i1 %0, %1
  %3 = icmp eq i32 %add14, 0
  %4 = or i1 %3, %2
  %5 = select i1 %4, i32 1, i32 %add14
  %div = sdiv i32 %ret.1, %5
  %cmp15 = icmp slt i32 %n, 11
  br i1 %cmp15, label %k, label %h

if.end20:                                         ; preds = %if.else, %for.end
  %ret.3 = phi i32 [ %ret.2, %for.end ], [ %ret.1, %if.else ]
  %mul21 = mul nsw i32 %ret.3, %n
  %and = and i32 %n, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %h, label %j

h:                                                ; preds = %if.end20, %if.then12
  %ret.4 = phi i32 [ %div, %if.then12 ], [ %mul21, %if.end20 ]
  %inc24 = add nsw i32 %ret.4, 1
  br label %while.body2

j:                                                ; preds = %if.end20
  %mul25 = mul i32 %n, 2
  %add26 = add nsw i32 %mul25, 20
  %add27 = add nsw i32 %add26, %mul21
  br label %l

k:                                                ; preds = %if.then12
  %mul28 = mul nsw i32 %div, %n
  br label %l

l:                                                ; preds = %k, %j
  %storemerge = phi i32 [ %add27, %j ], [ %mul28, %k ]
  %and29 = and i32 %n, 1
  %tobool30 = icmp eq i32 %and29, 0
  br i1 %tobool30, label %if.end33, label %if.then31

if.then31:                                        ; preds = %l
  br label %for.cond35

if.end33:                                         ; preds = %l
  br label %while.body

for.cond35:                                       ; preds = %for.body39, %if.then31
  %ret.5.in = phi i32 [ %storemerge, %if.then31 ], [ %ret.5, %for.body39 ]
  %storemerge1 = phi i32 [ 0, %if.then31 ], [ %inc42, %for.body39 ]
  %ret.5 = add nsw i32 %ret.5.in, 1
  %div36 = sdiv i32 %n, 4
  %cmp37 = icmp slt i32 %storemerge1, %div36
  br i1 %cmp37, label %for.body39, label %n46

for.body39:                                       ; preds = %for.cond35
  %inc42 = add nsw i32 %storemerge1, 1
  br label %for.cond35

i44:                                              ; preds = %for.end
  %6 = icmp eq i32 %ret.2, -2147483648
  %7 = icmp eq i32 %n, -1
  %8 = and i1 %7, %6
  %9 = icmp eq i32 %n, 0
  %10 = or i1 %9, %8
  %11 = select i1 %10, i32 1, i32 %n
  %div45 = sdiv i32 %ret.2, %11
  br label %n46

n46:                                              ; preds = %i44, %for.cond35
  %ret.6 = phi i32 [ %div45, %i44 ], [ %ret.5, %for.cond35 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.6, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization11, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization11
; CHECK: br i1 true, label %[[WHILEBODYUNIFORM:.+]], label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: br label %[[WHILEBODY2:.+]]

; CHECK: [[WHILEBODY2]]:
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[FORCONDPREHEADER:.+]], label %[[IFELSE:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[FOREND:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[FOREND]]:
; CHECK: br label %[[IFEND20:.+]]

; CHECK: [[IFELSE]]:
; CHECK: br label %[[IFTHEN12:.+]]

; CHECK: [[WHILEBODYUNIFORM]]:
; CHECK: br label %[[WHILEBODY2UNIFORM:.+]]

; CHECK: [[WHILEBODY2UNIFORM]]:
; CHECK: %[[CMPUNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMPUNIFORM]], label %[[FORCONDPREHEADERUNIFORM:.+]], label %[[IFELSEUNIFORM:.+]]

; CHECK: [[IFELSEUNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[IFEND20UNIFORM:.+]], label %[[IFELSEUNIFORMBOSCCINDIR:.+]]

; CHECK: [[IFTHEN12UNIFORM:.+]]:
; CHECK: %[[CMP15UNIFORM:cmp.+]] = icmp
; CHECK: br i1 %[[CMP15UNIFORM]], label %[[KUNIFORM:.+]], label %[[HUNIFORM:.+]]

; CHECK: [[FORCONDPREHEADERUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM:.+]]

; CHECK: [[FORCONDUNIFORM]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODYUNIFORM:.+]], label %[[FORENDUNIFORM:.+]]

; CHECK: [[FORBODYUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM]]

; CHECK: [[FORENDUNIFORM]]:
; CHECK: %[[CMP7UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP7UNIFORM]], label %[[I44UNIFORM:.+]], label %[[IFEND20UNIFORM]]

; CHECK: [[IFEND20UNIFORM]]:
; CHECK: %[[TOBOOLUNIFORM:.+]] = icmp
; CHECK: br i1 %[[TOBOOLUNIFORM]], label %[[HUNIFORM]], label %[[JUNIFORM:.+]]

; CHECK: [[IFELSEUNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN12UNIFORM]], label %[[IFELSEUNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFELSEUNIFORMBOSCCSTORE]]:
; CHECK: br label %[[IFTHEN12]]

; CHECK: [[HUNIFORM]]:
; CHECK: br label %[[WHILEBODY2UNIFORM]]

; CHECK: [[KUNIFORM]]:
; CHECK: br label %[[LUNIFORM:.+]]

; CHECK: [[JUNIFORM]]:
; CHECK: br label %[[LUNIFORM]]

; CHECK: [[LUNIFORM]]:
; CHECK: %[[TOBOOL30UNIFORM:.+]] = icmp
; CHECK: br i1 %[[TOBOOL30UNIFORM]], label %[[WHILEBODYUNIFORM]], label %[[FORCOND35PREHEADERUNIFORM:.+]]

; CHECK: [[FORCOND35PREHEADERUNIFORM]]:
; CHECK: br label %[[FORCOND35UNIFORM:.+]]

; CHECK: [[FORCOND35UNIFORM]]:
; CHECK: %[[CMP37UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP37UNIFORM]], label %[[FORBODY39UNIFORM:.+]], label %[[N46LOOPEXITUNIFORM:.+]]

; CHECK: [[FORBODY39UNIFORM]]:
; CHECK: br label %[[FORCOND35UNIFORM]]

; CHECK: [[N46LOOPEXITUNIFORM]]:
; CHECK: br label %[[N46UNIFORM:.+]]

; CHECK: [[I44UNIFORM]]:
; CHECK: br label %[[N46:.+]]

; CHECK: [[IFTHEN12]]:
; CHECK: br label %[[IFEND20]]

; CHECK: [[IFEND20]]:
; CHECK: br label %[[H:.+]]

; CHECK: [[H]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY2]], label %[[WHILEBODY2PUREEXIT:.+]]

; CHECK: [[WHILEBODY2PUREEXIT]]:
; CHECK: br label %[[K:.+]]

; CHECK: [[J:.+]]:
; CHECK: br label %[[L:.+]]

; CHECK: [[K]]:
; CHECK: br label %[[KELSE:.+]]

; CHECK: [[KELSE]]:
; CHECK: br label %[[J]]

; CHECK: [[L]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[FORCOND35PREHEADER:.+]]

; CHECK: [[FORCOND35PREHEADER]]:
; CHECK: br label %[[FORCOND35:.+]]

; CHECK: [[FORCOND35PREHEADERELSE:.+]]:
; CHECK: br label %[[I44:.+]]

; CHECK: [[FORCOND35]]:
; CHECK: %[[CMP37:.+]] = icmp
; CHECK: br i1 %[[CMP37]], label %[[FORBODY39:.+]], label %[[N46LOOPEXIT:.+]]

; CHECK: [[FORBODY39]]:
; CHECK: br label %[[FORCOND35]]

; CHECK: [[I44]]:
; CHECK: br label %[[N46]]

; CHECK: [[N46LOOPEXIT]]:
; CHECK: br label %[[FORCOND35PREHEADERELSE]]

; CHECK: [[N46]]:
; CHECK: ret void
