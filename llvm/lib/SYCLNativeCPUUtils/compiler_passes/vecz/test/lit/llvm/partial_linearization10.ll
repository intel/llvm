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

; RUN: veczc -k partial_linearization10 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;            a
;            |
;            b <-----.
;           / \      |
;          c   d     |
;         / \ /      |
;        /   e       |
;       /    |       |
;      /     g <---. |
;     /     / \    | |
;    /     h   i   | |
;   f     / \ / \  | |
;   |    j   k   l | |
;   |   /|  / \ /  | |
;   |  m | n   o --' |
;   | /  |/          |
;   |/   q ----------'
;   p    |
;    \   r
;     \ /
;      s
;
; * where nodes b, c, g, h, j, k and q are uniform branches, and node i is a
;   varying branch.
; * where nodes k, l, o, n, m, p, q, r and s are divergent.
;
; With partial linearization, it will be transformed as follows:
;
;          a
;          |
;          b <-----.
;         / \      |
;        c   d     |
;       / \ /      |
;      /   e       |
;     /    |       |
;    /     g <---. |
;   f     / \    | |
;   |    /   \   | |
;   |   h     i  | |
;   |  / \    |  | |
;   | j   |   l  | |
;   | |    \ /   | |
;   | |     k    | |
;   |  \    |    | |
;   |   \   o ---' |
;   |    \ /       |
;   |     n        |
;    \    |        |
;     \   q -------'
;      \ /
;       m
;       |
;       r
;       |
;       p
;       |
;       s
;
; __kernel void partial_linearization10(__global int *out, int n) {
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
;           // j
;           goto j;
;         }
;       } else {
;         // i
;         if (ret + id >= n) {
;           // l
;           ret /= n * n + ret;
;           goto o;
;         }
;       }
;       // k
;       if (n & 1) {
;         // n
;         ret += n * ret;
;         goto n;
;       }
;       // o
; o:
;       ret++;
;     }
; j:
;     if (n < 2) {
;       // m
;       ret += n * 2 + 20;
;       goto p;
;     } else {
;       goto q;
;     }
; n:
;     ret *= 4;
; q:
;     if (n & 1) {
;       // r
;       ret++;
;       goto r;
;     }
;   }
;
; r:
;   for (int i = 0; i < n / 4; i++) ret++;
;   goto s;
;
; f:
;   ret /= n;
;   goto p;
;
; p:
;   for (int i = 0; i < n * 2; i++) ret++;
;
; s:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization10(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end55, %entry
  %ret.0 = phi i32 [ 0, %entry ], [ %ret.5, %if.end55 ]
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.then
  %ret.1 = phi i32 [ %ret.0, %if.then ], [ %inc, %for.body ]
  %storemerge5 = phi i32 [ 0, %if.then ], [ %inc4, %for.body ]
  %mul = shl nsw i32 %n, 1
  %cmp2 = icmp slt i32 %storemerge5, %mul
  br i1 %cmp2, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %inc = add nsw i32 %ret.1, 1
  %inc4 = add nsw i32 %storemerge5, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %cmp5 = icmp slt i32 %n, 11
  br i1 %cmp5, label %f, label %if.end17

if.else:                                          ; preds = %while.body
  br label %for.cond9

for.cond9:                                        ; preds = %for.body12, %if.else
  %ret.2 = phi i32 [ %ret.0, %if.else ], [ %inc13, %for.body12 ]
  %storemerge = phi i32 [ 0, %if.else ], [ %inc15, %for.body12 ]
  %div = sdiv i32 %n, 4
  %cmp10 = icmp slt i32 %storemerge, %div
  br i1 %cmp10, label %for.body12, label %if.end17

for.body12:                                       ; preds = %for.cond9
  %inc13 = add nsw i32 %ret.2, 1
  %inc15 = add nsw i32 %storemerge, 1
  br label %for.cond9

if.end17:                                         ; preds = %for.cond9, %for.end
  %ret.3 = phi i32 [ %ret.1, %for.end ], [ %ret.2, %for.cond9 ]
  br label %while.body20

while.body20:                                     ; preds = %o, %if.end17
  %storemerge1.in = phi i32 [ %ret.3, %if.end17 ], [ %ret.4, %o ]
  %storemerge1 = add nsw i32 %storemerge1.in, 1
  %and = and i32 %n, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.else26, label %if.then21

if.then21:                                        ; preds = %while.body20
  %cmp22 = icmp slt i32 %n, 3
  br i1 %cmp22, label %j, label %if.end34

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
  br label %o

if.end34:                                         ; preds = %if.else26, %if.then21
  %and35 = and i32 %n, 1
  %tobool36 = icmp eq i32 %and35, 0
  br i1 %tobool36, label %o, label %if.then37

if.then37:                                        ; preds = %if.end34
  %mul38 = mul nsw i32 %storemerge1, %n
  %add39 = add nsw i32 %mul38, %storemerge1
  %mul50 = shl nsw i32 %add39, 2
  br label %q

o:                                                ; preds = %if.end34, %if.then29
  %ret.4 = phi i32 [ %div32, %if.then29 ], [ %storemerge1, %if.end34 ]
  br label %while.body20

j:                                                ; preds = %if.then21
  %cmp42 = icmp eq i32 %n, 2
  br i1 %cmp42, label %q, label %if.then44

if.then44:                                        ; preds = %j
  %mul45 = mul i32 %n, 2
  %add46 = add nsw i32 %mul45, 20
  %add47 = add nsw i32 %add46, %storemerge1
  br label %p

q:                                                ; preds = %j, %if.then37
  %ret.5 = phi i32 [ %mul50, %if.then37 ], [ %storemerge1, %j ]
  %and51 = and i32 %n, 1
  %tobool52 = icmp eq i32 %and51, 0
  br i1 %tobool52, label %if.end55, label %if.then53

if.then53:                                        ; preds = %q
  br label %for.cond57

if.end55:                                         ; preds = %q
  br label %while.body

for.cond57:                                       ; preds = %for.body61, %if.then53
  %ret.6.in = phi i32 [ %ret.5, %if.then53 ], [ %ret.6, %for.body61 ]
  %storemerge2 = phi i32 [ 0, %if.then53 ], [ %inc64, %for.body61 ]
  %ret.6 = add nsw i32 %ret.6.in, 1
  %div58 = sdiv i32 %n, 4
  %cmp59 = icmp slt i32 %storemerge2, %div58
  br i1 %cmp59, label %for.body61, label %s

for.body61:                                       ; preds = %for.cond57
  %inc64 = add nsw i32 %storemerge2, 1
  br label %for.cond57

f:                                                ; preds = %for.end
  %2 = icmp eq i32 %ret.1, -2147483648
  %3 = icmp eq i32 %n, -1
  %4 = and i1 %3, %2
  %5 = icmp eq i32 %n, 0
  %6 = or i1 %5, %4
  %7 = select i1 %6, i32 1, i32 %n
  %div66 = sdiv i32 %ret.1, %7
  br label %p

p:                                                ; preds = %f, %if.then44
  %storemerge3 = phi i32 [ %add47, %if.then44 ], [ %div66, %f ]
  br label %for.cond68

for.cond68:                                       ; preds = %for.body72, %p
  %ret.7 = phi i32 [ %storemerge3, %p ], [ %inc73, %for.body72 ]
  %storemerge4 = phi i32 [ 0, %p ], [ %inc75, %for.body72 ]
  %mul69 = shl nsw i32 %n, 1
  %cmp70 = icmp slt i32 %storemerge4, %mul69
  br i1 %cmp70, label %for.body72, label %s

for.body72:                                       ; preds = %for.cond68
  %inc73 = add nsw i32 %ret.7, 1
  %inc75 = add nsw i32 %storemerge4, 1
  br label %for.cond68

s:                                                ; preds = %for.cond68, %for.cond57
  %ret.8 = phi i32 [ %ret.6, %for.cond57 ], [ %ret.7, %for.cond68 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.8, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization10, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization10
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
; CHECK: %[[AND:.+]] = and i32
; CHECK: %[[TOBOOL:.+]] = icmp eq i32 %[[AND]]
; CHECK: br i1 %[[TOBOOL]], label %[[IFELSE26:.+]], label %[[IFTHEN21:.+]]

; CHECK: [[IFTHEN21]]:
; CHECK: %[[CMP22:.+]] = icmp
; CHECK: br i1 %[[CMP22]], label %[[J:.+]], label %[[IFEND34:.+]]

; CHECK: [[IFELSE26]]:
; CHECK: br label %[[IFTHEN29:.+]]

; CHECK: [[IFTHEN29]]:
; CHECK: br label %[[IFEND34]]

; CHECK: [[IFEND34]]:
; CHECK: br label %[[O:.+]]

; CHECK: [[IFTHEN37:.+]]:
; CHECK: br label %[[IFTHEN37ELSE:.+]]

; CHECK: [[IFTHEN37ELSE]]:
; CHECK: br i1 %{{.+}}, label %[[JELSE:.+]], label %[[JSPLIT:.+]]

; CHECK: [[O]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY20]], label %[[WHILEBODY20PUREEXIT:.+]]

; CHECK: [[WHILEBODY20PUREEXIT]]:
; CHECK: br label %[[IFTHEN37]]

; CHECK: [[J]]:
; CHECK: br label %[[WHILEBODY20PUREEXIT]]

; CHECK: [[JELSE]]:
; CHECK: br label %[[Q:.+]]

; CHECK: [[JSPLIT]]:
; CHECK: br label %[[Q]]

; CHECK: [[IFTHEN44:.+]]:
; CHECK: br label %[[IFTHEN44ELSE:.+]]

; CHECK: [[IFTHEN44ELSE]]:
; CHECK: br label %[[FORCOND57PREHEADER:.+]]

; CHECK: [[Q]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[IFTHEN44]]

; CHECK: [[FORCOND57PREHEADER]]:
; CHECK: br label %[[FORCOND57:.+]]

; CHECK: [[FORCOND57PREHEADERELSE:.+]]:
; CHECK: br i1 %{{.+}}, label %[[FELSE:.+]], label %[[FSPLIT:.+]]

; CHECK: [[FORCOND57]]:
; CHECK: %[[CMP59:.+]] = icmp
; CHECK: br i1 %[[CMP59]], label %[[FORBODY61:.+]], label %[[SLOOPEXIT2:.+]]

; CHECK: [[FORBODY61]]:
; CHECK: br label %[[FORCOND57]]

; CHECK: [[F]]:
; CHECK: br label %[[WHILEBODYPUREEXIT]]

; CHECK: [[FELSE]]:
; CHECK: br label %[[P:.+]]

; CHECK: [[FSPLIT]]:
; CHECK: br label %[[P]]

; CHECK: [[P]]:
; CHECK: br label %[[FORCOND68:.+]]

; CHECK: [[FORCOND68]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY72:.+]], label %[[SLOOPEXIT:.+]]

; CHECK: [[FORBODY72]]:
; CHECK: br label %[[FORCOND68]]

; CHECK: [[SLOOPEXIT]]:
; CHECK: br label %[[S:.+]]

; CHECK: [[SLOOPEXIT2]]:
; CHECK: br label %[[FORCOND57PREHEADERELSE]]

; CHECK: [[S]]:
; CHECK: ret void
