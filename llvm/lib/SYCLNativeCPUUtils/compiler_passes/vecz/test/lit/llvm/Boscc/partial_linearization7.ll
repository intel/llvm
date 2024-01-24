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

; RUN: veczc -k partial_linearization7 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;       a
;      / \
;     b   c
;    / \ / \
;   d   e   f
;    \ / \ /
;     g   h
;      \ /
;       i
;
; * where nodes a, c and e are uniform branches, and node b is a varying
;   branch.
; * where nodes d, e, g and i are divergent.
;
; With BOSCC, it will be transformed as follows:
;
;          a
;         / \
;        /   \
;       /     \
;      /       \
;     b____     c
;    / \   \   / \
;   d   e   d'|   |
;    \ / \   \|   |
;     g   h   e'  f
;      \ /     \ /
;       i       h'
;       |       |
;       |       g'
;       |       |
;       |       i'
;        \     /
;         \   /
;          \ /
;           &
;
; where '&' represents merge blocks of BOSCC regions.
;
; __kernel void partial_linearization7(__global int *out, int n) {
;   int id = get_global_id(0);
;   int i = 0;
;
;   if (n > 10) { // a
;     if (n + id > 10) { // b
;       i = n * 10; // d
;       goto g;
;     } else {
;       goto e;
;     }
;   } else {
;     if (n < 5) { // c
;       goto e;
;     } else {
;       for (int j = 0; j < n; j++) { i++; }
;       goto h;
;     }
;   }
;
; e:
;   if (n > 5) {
;     goto g;
;   } else {
;     i = n * 3 / 5;
;     goto h;
;   }
;
; g:
;   for (int j = 0; j < n; j++) { i++; }
;   goto i;
;
; h:
;   i = n + id / 3;
;
; i:
;   out[id] = i;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization7(i32 addrspace(1)* %out, i32 noundef %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  %cmp = icmp sgt i32 %n, 10
  br i1 %cmp, label %if.then, label %if.else5

if.then:                                          ; preds = %entry
  %add = add nsw i32 %conv, %n
  %cmp2 = icmp sgt i32 %add, 10
  br i1 %cmp2, label %if.then4, label %e

if.then4:                                         ; preds = %if.then
  %mul = mul nsw i32 %n, 10
  br label %g

if.else5:                                         ; preds = %entry
  %cmp6 = icmp slt i32 %n, 5
  br i1 %cmp6, label %e, label %if.else9

if.else9:                                         ; preds = %if.else5
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.else9
  %storemerge = phi i32 [ 0, %if.else9 ], [ %inc12, %for.body ]
  %cmp10 = icmp slt i32 %storemerge, %n
  br i1 %cmp10, label %for.body, label %h

for.body:                                         ; preds = %for.cond
  %inc12 = add nsw i32 %storemerge, 1
  br label %for.cond

e:                                                ; preds = %if.else5, %if.then
  %cmp13 = icmp sgt i32 %n, 5
  br i1 %cmp13, label %g, label %h

g:                                                ; preds = %e, %if.then4
  %i.1 = phi i32 [ %mul, %if.then4 ], [ 0, %e ]
  br label %for.cond19

for.cond19:                                       ; preds = %for.body22, %g
  %i.2 = phi i32 [ %i.1, %g ], [ %inc23, %for.body22 ]
  %storemerge1 = phi i32 [ 0, %g ], [ %inc25, %for.body22 ]
  %cmp20 = icmp slt i32 %storemerge1, %n
  br i1 %cmp20, label %for.body22, label %i29

for.body22:                                       ; preds = %for.cond19
  %inc23 = add nsw i32 %i.2, 1
  %inc25 = add nsw i32 %storemerge1, 1
  br label %for.cond19

h:                                                ; preds = %e, %for.cond
  %div27 = sdiv i32 %conv, 3
  %add28 = add nsw i32 %div27, %n
  br label %i29

i29:                                              ; preds = %h, %for.cond19
  %i.3 = phi i32 [ %add28, %h ], [ %i.2, %for.cond19 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %i.3, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization7, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization7
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[IFTHEN:.+]], label %[[IFELSE5:.+]]

; CHECK: [[IFTHEN]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN4UNIFORM:.+]], label %[[IFTHENBOSCCINDIR:.+]]

; CHECK: [[IFTHEN4UNIFORM]]:
; CHECK: br label %[[GUNIFORM:.+]]

; CHECK: [[IFTHENBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[EUNIFORM:.+]], label %[[IFTHEN4:.+]]

; CHECK: [[EUNIFORM]]:
; CHECK: %[[CMP13UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP13UNIFORM]], label %[[GUNIFORM]], label %[[HUNIFORM:.+]]

; CHECK: [[HUNIFORM]]:
; CHECK: br label %[[I29UNIFORM:.+]]

; CHECK: [[GUNIFORM]]:
; CHECK: br label %[[FORCOND19UNIFORM:.+]]

; CHECK: [[FORCOND19UNIFORM]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY22UNIFORM:.+]], label %[[I29LOOPEXITUNIFORM:.+]]

; CHECK: [[FORBODY22UNIFORM]]:
; CHECK: br label %[[FORCOND19UNIFORM]]

; CHECK: [[I29LOOPEXITUNIFORM]]:
; CHECK: br label %[[I29:.+]]

; CHECK: [[IFTHEN4]]:
; CHECK: br label %[[E:.+]]

; CHECK: [[IFELSE5]]:
; CHECK: %[[CMP6:.+]] = icmp
; CHECK: br i1 %[[CMP6]], label %[[E]], label %[[FORCONDPREHEADER:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[HLOOPEXIT:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[E]]:
; CHECK: %[[CMP13:.+]] = icmp
; CHECK: br i1 %[[CMP13]], label %[[G:.+]], label %[[H:.+]]

; CHECK: [[G]]:
; CHECK: br label %[[FORCOND19:.+]]

; CHECK: [[FORCOND19]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY22:.+]], label %[[I29LOOPEXIT:.+]]

; CHECK: [[FORBODY22]]:
; CHECK: br label %[[FORCOND19]]

; CHECK: [[HLOOPEXIT]]:
; CHECK: br label %[[H]]

; CHECK: [[H]]:
; CHECK: br label %[[G]]

; CHECK: [[I29LOOPEXIT]]:
; CHECK: br label %[[I29]]

; CHECK: [[I29]]:
; CHECK: ret void
