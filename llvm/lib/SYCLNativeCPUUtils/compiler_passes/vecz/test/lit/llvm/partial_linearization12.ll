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

; RUN: veczc -k partial_linearization12 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;              a
;              |
;              b <-----.
;             / \      |
;            c   d     |
;           / \ /      |
;          /   e       |
;         /    |       |
;        /     g <---. |
;       f     / \    | |
;       |    h   i   | |
;       |   /   / \  | |
;       |  /   k   l | |
;       | /    |\ /| | |
;       |/     |/ \| | |
;       j      m   n | |
;      /|     / \ /  | |
;     / |    o   p --' |
;    /  |   /   /      |
;   |   |  /   r       |
;   |   | /    |       |
;   |   |/     s ------'
;   |   |     /
;   |  /|    t
;   | / |   /
;   |/  |  /
;   q   | /
;   |   |/
;   |   u
;    \ /
;     v
;
; * where nodes b, c, g, j, k, l, m, p and s are uniform branches,
;   and node i is a varying branch.
; * where nodes k, l, o, n, m, p, q, s, r, t and v are divergent.
;
; With partial linearization, it will be transformed as follows:
;
;         a
;         |
;         b <----.
;        / \     |
;       c   d    |
;      / \ /     |
;     /   e      |
;    /    |      |
;   f     g <--. |
;   |    / \   | |
;   |   h   i  | |
;   |  /    |  | |
;   | /     l  | |
;   |/      |  | |
;   j       k  | |
;   |\      |  | |
;   | \     n  | |
;   |  \    |  | |
;   |   |   m  | |
;   |   |   |  | |
;   |   |   p -' |
;   |   |  /     |
;   |   | r      |
;   |   | |      |
;   |   | s -----'
;   |   |/
;   |   o
;   |  /
;   | t
;   |/
;   u
;   |
;   q
;   |
;   v
;
; __kernel void partial_linearization12(__global int *out, int n) {
;   // a
;   int id = get_global_id(0);
;   int ret = 0;
;
;   while (1) {
;     if (n > 0) { // b
;       // c
;       for (int i = 0; i < n * 2; i++) ret++;
;       if (n < 5) {
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
;       if (n <= 2) { // g
;         // h
;         ret -= n * ret;
;         for (int i = 0; i < n * 2; i++) ret++;
;         // j
;         goto j;
;       } else {
;         // i
;         if (ret + id >= n) {
;           // k
;           ret /= n * n + ret;
;           if (n < 5) {
;             // m
;             ret -= n;
;             goto m;
;           } else {
;             // n
;             ret += n;
;             goto n;
;           }
;         } else {
;           // l
;           if (n >= 5) {
;             // m
;             ret += n;
;             goto m;
;           } else {
;             // n
;             ret -= n;
;             goto n;
;           }
;         }
;       }
;       // m
; m:
;       if (n & 1) {
;         // o
;         ret *= n;
;         goto q;
;       } else {
;         // p
;         goto p;
;       }
;
;       // n
; n:
;       ret *= ret;
;       // p
; p:
;       if (n > 3) {
;         goto r;
;       }
;       ret++;
;     }
;
;     // r
; r:
;     ret *= 4;
;     for (int i = 0; i < n / 4; i++) ret++;
;
;     // s
;     if (n & 1) {
;       goto t;
;     }
;     ret++;
;   }
;
; f:
;   ret /= n;
;   goto j;
;
; j:
;   if (n == 2) {
;     goto q;
;   } else {
;     goto u;
;   }
;
; t:
;   for (int i = 0; i < n + 1; i++) ret++;
;   goto u;
;
; q:
;   for (int i = 0; i < n / 4; i++) ret++;
;   goto v;
;
; u:
;   for (int i = 0; i < n * 2; i++) ret++;
;
; v:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization12(i32 addrspace(1)* %out, i32 noundef %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end79, %entry
  %storemerge = phi i32 [ 0, %entry ], [ %inc80, %if.end79 ]
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.then
  %ret.0 = phi i32 [ %storemerge, %if.then ], [ %inc, %for.body ]
  %storemerge10 = phi i32 [ 0, %if.then ], [ %inc4, %for.body ]
  %mul = shl nsw i32 %n, 1
  %cmp2 = icmp slt i32 %storemerge10, %mul
  br i1 %cmp2, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %inc = add nsw i32 %ret.0, 1
  %inc4 = add nsw i32 %storemerge10, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %cmp5 = icmp slt i32 %n, 5
  br i1 %cmp5, label %f, label %if.end17

if.else:                                          ; preds = %while.body
  br label %for.cond9

for.cond9:                                        ; preds = %for.body12, %if.else
  %ret.1 = phi i32 [ %storemerge, %if.else ], [ %inc13, %for.body12 ]
  %storemerge1 = phi i32 [ 0, %if.else ], [ %inc15, %for.body12 ]
  %div = sdiv i32 %n, 4
  %cmp10 = icmp slt i32 %storemerge1, %div
  br i1 %cmp10, label %for.body12, label %if.end17

for.body12:                                       ; preds = %for.cond9
  %inc13 = add nsw i32 %ret.1, 1
  %inc15 = add nsw i32 %storemerge1, 1
  br label %for.cond9

if.end17:                                         ; preds = %for.cond9, %for.end
  %ret.2 = phi i32 [ %ret.0, %for.end ], [ %ret.1, %for.cond9 ]
  br label %while.body20

while.body20:                                     ; preds = %if.end63, %if.end17
  %storemerge2.in = phi i32 [ %ret.2, %if.end17 ], [ %ret.4, %if.end63 ]
  %storemerge2 = add nsw i32 %storemerge2.in, 1
  %cmp21 = icmp slt i32 %n, 3
  br i1 %cmp21, label %if.then23, label %if.else35

if.then23:                                        ; preds = %while.body20
  %mul24 = mul nsw i32 %storemerge2, %n
  %sub = sub nsw i32 %storemerge2, %mul24
  br label %for.cond26

for.cond26:                                       ; preds = %for.body30, %if.then23
  %ret.3 = phi i32 [ %sub, %if.then23 ], [ %inc31, %for.body30 ]
  %storemerge9 = phi i32 [ 0, %if.then23 ], [ %inc33, %for.body30 ]
  %mul27 = shl nsw i32 %n, 1
  %cmp28 = icmp slt i32 %storemerge9, %mul27
  br i1 %cmp28, label %for.body30, label %j

for.body30:                                       ; preds = %for.cond26
  %inc31 = add nsw i32 %ret.3, 1
  %inc33 = add nsw i32 %storemerge9, 1
  br label %for.cond26

if.else35:                                        ; preds = %while.body20
  %add = add nsw i32 %storemerge2, %conv
  %cmp36 = icmp slt i32 %add, %n
  br i1 %cmp36, label %if.else48, label %if.then38

if.then38:                                        ; preds = %if.else35
  %mul39 = mul nsw i32 %n, %n
  %add40 = add nsw i32 %storemerge2, %mul39
  %0 = icmp eq i32 %add40, 0
  %1 = select i1 %0, i32 1, i32 %add40
  %div41 = sdiv i32 %storemerge2, %1
  %cmp42 = icmp slt i32 %n, 5
  br i1 %cmp42, label %if.then44, label %if.else46

if.then44:                                        ; preds = %if.then38
  %sub45 = sub nsw i32 %div41, %n
  br label %m

if.else46:                                        ; preds = %if.then38
  %add47 = add nsw i32 %div41, %n
  br label %n58

if.else48:                                        ; preds = %if.else35
  %cmp49 = icmp sgt i32 %n, 4
  br i1 %cmp49, label %if.then51, label %if.else53

if.then51:                                        ; preds = %if.else48
  %add52 = add nsw i32 %storemerge2, %n
  br label %m

if.else53:                                        ; preds = %if.else48
  %sub54 = sub nsw i32 %storemerge2, %n
  br label %n58

m:                                                ; preds = %if.then51, %if.then44
  %storemerge7 = phi i32 [ %add52, %if.then51 ], [ %sub45, %if.then44 ]
  %and = and i32 %n, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %p, label %if.then55

if.then55:                                        ; preds = %m
  %mul56 = mul nsw i32 %storemerge7, %n
  br label %q

n58:                                              ; preds = %if.else53, %if.else46
  %storemerge3 = phi i32 [ %sub54, %if.else53 ], [ %add47, %if.else46 ]
  %mul59 = mul nsw i32 %storemerge3, %storemerge3
  br label %p

p:                                                ; preds = %n58, %m
  %ret.4 = phi i32 [ %mul59, %n58 ], [ %storemerge7, %m ]
  %cmp60 = icmp sgt i32 %n, 3
  br i1 %cmp60, label %r, label %if.end63

if.end63:                                         ; preds = %p
  br label %while.body20

r:                                                ; preds = %p
  %mul65 = shl nsw i32 %ret.4, 2
  br label %for.cond67

for.cond67:                                       ; preds = %for.body71, %r
  %ret.5 = phi i32 [ %mul65, %r ], [ %inc72, %for.body71 ]
  %storemerge4 = phi i32 [ 0, %r ], [ %inc74, %for.body71 ]
  %div68 = sdiv i32 %n, 4
  %cmp69 = icmp slt i32 %storemerge4, %div68
  br i1 %cmp69, label %for.body71, label %for.end75

for.body71:                                       ; preds = %for.cond67
  %inc72 = add nsw i32 %ret.5, 1
  %inc74 = add nsw i32 %storemerge4, 1
  br label %for.cond67

for.end75:                                        ; preds = %for.cond67
  %and76 = and i32 %n, 1
  %tobool77 = icmp eq i32 %and76, 0
  br i1 %tobool77, label %if.end79, label %t

if.end79:                                         ; preds = %for.end75
  %inc80 = add nsw i32 %ret.5, 1
  br label %while.body

f:                                                ; preds = %for.end
  %2 = icmp eq i32 %n, 0
  %3 = select i1 %2, i32 1, i32 %n
  %div81 = sdiv i32 %ret.0, %3
  br label %j

j:                                                ; preds = %f, %for.cond26
  %ret.6 = phi i32 [ %div81, %f ], [ %ret.3, %for.cond26 ]
  %cmp82 = icmp eq i32 %n, 2
  br i1 %cmp82, label %q, label %u

t:                                                ; preds = %for.end75
  br label %for.cond87

for.cond87:                                       ; preds = %for.body91, %t
  %ret.7 = phi i32 [ %ret.5, %t ], [ %inc92, %for.body91 ]
  %storemerge5 = phi i32 [ 0, %t ], [ %inc94, %for.body91 ]
  %cmp89 = icmp sgt i32 %storemerge5, %n
  br i1 %cmp89, label %u, label %for.body91

for.body91:                                       ; preds = %for.cond87
  %inc92 = add nsw i32 %ret.7, 1
  %inc94 = add nsw i32 %storemerge5, 1
  br label %for.cond87

q:                                                ; preds = %j, %if.then55
  %ret.8 = phi i32 [ %mul56, %if.then55 ], [ %ret.6, %j ]
  br label %for.cond97

for.cond97:                                       ; preds = %for.body101, %q
  %ret.9 = phi i32 [ %ret.8, %q ], [ %inc102, %for.body101 ]
  %storemerge8 = phi i32 [ 0, %q ], [ %inc104, %for.body101 ]
  %div98 = sdiv i32 %n, 4
  %cmp99 = icmp slt i32 %storemerge8, %div98
  br i1 %cmp99, label %for.body101, label %v

for.body101:                                      ; preds = %for.cond97
  %inc102 = add nsw i32 %ret.9, 1
  %inc104 = add nsw i32 %storemerge8, 1
  br label %for.cond97

u:                                                ; preds = %for.cond87, %j
  %ret.10 = phi i32 [ %ret.6, %j ], [ %ret.7, %for.cond87 ]
  br label %for.cond107

for.cond107:                                      ; preds = %for.body111, %u
  %ret.11 = phi i32 [ %ret.10, %u ], [ %inc112, %for.body111 ]
  %storemerge6 = phi i32 [ 0, %u ], [ %inc114, %for.body111 ]
  %mul108 = shl nsw i32 %n, 1
  %cmp109 = icmp slt i32 %storemerge6, %mul108
  br i1 %cmp109, label %for.body111, label %v

for.body111:                                      ; preds = %for.cond107
  %inc112 = add nsw i32 %ret.11, 1
  %inc114 = add nsw i32 %storemerge6, 1
  br label %for.cond107

v:                                                ; preds = %for.cond107, %for.cond97
  %ret.12 = phi i32 [ %ret.9, %for.cond97 ], [ %ret.11, %for.cond107 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.12, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization12, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization12
; CHECK: br label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[FORCONDPREHEADER:.+]], label %[[FORCOND9PREHEADER:.+]]

; CHECK: [[FORCOND9PREHEADER]]:
; CHECK: br label %[[FORCOND9:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[FOREND:.+]]

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
; CHECK: %[[CMP21:.+]] = icmp
; CHECK: br i1 %[[CMP21]], label %[[IFTHEN23:.+]], label %[[IFELSE35:.+]]

; CHECK: [[IFTHEN23]]:
; CHECK: br label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[IFTHEN23ELSE:.+]]:
; CHECK: br i1 %{{.+}}, label %[[FELSE:.+]], label %[[FSPLIT:.+]]

; CHECK: [[IFTHEN23SPLIT:.+]]:
; CHECK: br label %[[FORCOND26:.+]]

; CHECK: [[FORCOND26]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY30:.+]], label %[[JLOOPEXIT:.+]]

; CHECK: [[FORBODY30]]:
; CHECK: br label %[[FORCOND26]]

; CHECK: [[IFELSE35]]:
; CHECK: br label %[[IFTHEN38:.+]]

; CHECK: [[IFTHEN38]]:
; CHECK: %[[CMP42:.+]] = icmp slt i32
; CHECK: br i1 %[[CMP42]], label %[[IFTHEN44:.+]], label %[[IFELSE46:.+]]

; CHECK: [[IFTHEN44]]:
; CHECK: br label %[[IFELSE48:.+]]

; CHECK: [[IFELSE46]]:
; CHECK: br label %[[IFELSE48]]

; CHECK: [[IFELSE48]]:
; CHECK: %[[CMP49:.+]] = icmp
; CHECK: br i1 %[[CMP49]], label %[[IFTHEN51:.+]], label %[[IFELSE53:.+]]

; CHECK: [[IFTHEN51]]:
; CHECK: br label %[[N58:.+]]

; CHECK: [[IFELSE53]]:
; CHECK: br label %[[N58]]

; CHECK: [[M:.+]]:
; CHECK: br label %[[P:.+]]

; CHECK: [[IFTHEN55:.+]]:
; CHECK: br label %[[IFTHEN55ELSE:.+]]

; CHECK: [[IFTHEN55ELSE]]:
; CHECK: br label %[[FORCOND87PREHEADER:.+]]

; CHECK: [[N58]]:
; CHECK: br label %[[M]]

; CHECK: [[P]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY20]], label %[[WHILEBODY20PUREEXIT:.+]]

; CHECK: [[WHILEBODY20PUREEXIT]]:
; CHECK: br label %[[R:.+]]

; CHECK: [[R]]:
; CHECK: br label %[[FORCOND67:.+]]

; CHECK: [[FORCOND67]]:
; CHECK: %[[CMP69:.+]] = icmp
; CHECK: br i1 %[[CMP69]], label %[[FORBODY71:.+]], label %[[FOREND75:.+]]

; CHECK: [[FORBODY71]]:
; CHECK: br label %[[FORCOND67]]

; CHECK: [[FOREND75]]:
; CHECK: br label %[[IFEND79:.+]]

; CHECK: [[FORCOND87PREHEADER]]:
; CHECK: br label %[[FORCOND87:.+]]

; CHECK: [[FORCOND87PREHEADERELSE:.+]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN23ELSE]], label %[[IFTHEN23SPLIT]]

; CHECK: [[IFEND79]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[IFTHEN55]]

; CHECK: [[F]]:
; CHECK: br label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[FELSE]]:
; CHECK: br label %[[U:.+]]

; CHECK: [[FSPLIT]]:
; CHECK: br label %[[J:.+]]

; CHECK: [[JLOOPEXIT]]:
; CHECK: br label %[[J]]

; CHECK: [[J]]:
; CHECK: %[[CMP82:.+]] = icmp
; CHECK: br i1 %[[CMP82]], label %[[Q:.+]], label %[[U]]

; CHECK: [[FORCOND87]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(true)}}, label %[[ULOOPEXIT:.+]], label %[[FORBODY91:.+]]

; CHECK: [[FORBODY91]]:
; CHECK: br label %[[FORCOND87]]

; CHECK: [[Q]]:
; CHECK: br label %[[FORCOND97:.+]]

; CHECK: [[FORCOND97]]:
; CHECK: %[[CMP99:.+]] = icmp
; CHECK: br i1 %[[CMP99]], label %[[FORBODY101:.+]], label %[[VLOOPEXIT:.+]]

; CHECK: [[FORBODY101]]:
; CHECK: br label %[[FORCOND97]]

; CHECK: [[ULOOPEXIT]]:
; CHECK: br label %[[FORCOND87PREHEADERELSE]]

; CHECK: [[U]]:
; CHECK: br label %[[FORCOND107:.+]]

; CHECK: [[FORCOND107]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY111:.+]], label %[[VLOOPEXIT2:.+]]

; CHECK: [[FORBODY111]]:
; CHECK: br label %[[FORCOND107]]

; CHECK: [[VLOOPEXIT]]:
; CHECK: br label %[[V:.+]]

; CHECK: [[VLOOPEXIT2]]:
; CHECK: br label %[[Q]]

; CHECK: [[V]]:
; CHECK: ret void
