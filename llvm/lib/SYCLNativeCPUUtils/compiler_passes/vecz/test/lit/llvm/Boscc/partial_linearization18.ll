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

; RUN: veczc -k partial_linearization18 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;       a
;       |
;       b <--.
;      / \   |
;     c   d -'
;    / \  |
;   e   f |
;   |    \|
;   |     g
;   |    /
;   |   h
;    \ / \
;     i   j
;      \ /
;       k
;
; * where nodes b, and h are uniform branches, and nodes c and d are varying
;   branches.
; * where nodes e, f, g, i and k are divergent.
;
; With BOSCC, it will be transformed as follows:
;
;       a
;       |
;       b <--. .-> b' <--.
;      / \   | |  / \    |
;     c   d -' | c'  d' -'
;    / \__|\___' |   |
;   e   f |`---> f'  |
;   |    \|      |   |
;   |     g      e'  |
;   |    /        \ /
;   |   h          g'
;    \ / \         |
;     i   j        h'
;      \ /        / \
;       k        |   j'
;       |         \ /
;       |          i'
;       |          |
;       `--> & <-- k'
;
; where '&' represents merge blocks of BOSCC regions.
;
; __kernel void partial_linearization18(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;   int i = 0;
;
;   while (1) {
;     if (n > 5) {
;       if (id + i % 2 == 0) {
;         goto e;
;       } else {
;         goto f;
;       }
;     }
;     if (++i + id > 3) {
;       goto g;
;     }
;   }
;
; f:
;   for (int i = 0; i < n + 5; i++) ret += 2;
;   goto g;
;
; g:
;   for (int i = 1; i < n * 2; i++) ret *= i;
;   goto h;
;
; e:
;   for (int i = 0; i < n + 5; i++) ret++;
;   goto i;
;
; h:
;   if (n > 3) {
; i:
;     ret++;
;   } else {
;     ret *= 3;
;   }
;
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization18(i32 addrspace(1)* %out, i32 noundef %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %cmp = icmp sgt i32 %n, 5
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  %rem = and i32 %i.0, 1
  %add = sub nsw i32 0, %rem
  %cmp2 = icmp eq i32 %conv, %add
  br i1 %cmp2, label %for.cond26, label %for.cond

if.end:                                           ; preds = %while.body
  %inc = add nuw nsw i32 %i.0, 1
  %add5 = add nsw i32 %inc, %conv
  %cmp6 = icmp sgt i32 %add5, 3
  br i1 %cmp6, label %g, label %while.body

for.cond:                                         ; preds = %for.body, %if.then
  %ret.0 = phi i32 [ %add14, %for.body ], [ 0, %if.then ]
  %storemerge2 = phi i32 [ %inc15, %for.body ], [ 0, %if.then ]
  %add11 = add nsw i32 %n, 5
  %cmp12 = icmp slt i32 %storemerge2, %add11
  br i1 %cmp12, label %for.body, label %g

for.body:                                         ; preds = %for.cond
  %add14 = add nuw nsw i32 %ret.0, 2
  %inc15 = add nuw nsw i32 %storemerge2, 1
  br label %for.cond

g:                                                ; preds = %for.cond, %if.end
  %ret.1 = phi i32 [ 0, %if.end ], [ %ret.0, %for.cond ]
  br label %for.cond17

for.cond17:                                       ; preds = %for.body20, %g
  %ret.2 = phi i32 [ %ret.1, %g ], [ %mul21, %for.body20 ]
  %storemerge = phi i32 [ 1, %g ], [ %inc23, %for.body20 ]
  %mul = shl nsw i32 %n, 1
  %cmp18 = icmp slt i32 %storemerge, %mul
  br i1 %cmp18, label %for.body20, label %h

for.body20:                                       ; preds = %for.cond17
  %mul21 = mul nsw i32 %storemerge, %ret.2
  %inc23 = add nuw nsw i32 %storemerge, 1
  br label %for.cond17

for.cond26:                                       ; preds = %for.body30, %if.then
  %ret.3 = phi i32 [ %inc31, %for.body30 ], [ 0, %if.then ]
  %storemerge3 = phi i32 [ %inc33, %for.body30 ], [ 0, %if.then ]
  %add27 = add nsw i32 %n, 5
  %cmp28 = icmp slt i32 %storemerge3, %add27
  br i1 %cmp28, label %for.body30, label %i38

for.body30:                                       ; preds = %for.cond26
  %inc31 = add nuw nsw i32 %ret.3, 1
  %inc33 = add nuw nsw i32 %storemerge3, 1
  br label %for.cond26

h:                                                ; preds = %for.cond17
  %cmp35 = icmp sgt i32 %n, 3
  br i1 %cmp35, label %i38, label %if.else40

i38:                                              ; preds = %h, %for.cond26
  %ret.4 = phi i32 [ %ret.3, %for.cond26 ], [ %ret.2, %h ]
  %inc39 = add nsw i32 %ret.4, 1
  br label %if.end42

if.else40:                                        ; preds = %h
  %mul41 = mul nsw i32 %ret.2, 3
  br label %if.end42

if.end42:                                         ; preds = %if.else40, %i38
  %storemerge1 = phi i32 [ %mul41, %if.else40 ], [ %inc39, %i38 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %storemerge1, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization18, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization18
; CHECK: br i1 true, label %[[WHILEBODYUNIFORM:.+]], label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[IFTHEN:.+]], label %[[IFEND:.+]]

; CHECK: [[IFTHEN]]:
; CHECK: br label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[IFTHENELSE:.+]]:
; CHECK: br label %[[G:.+]]

; CHECK: [[IFTHENSPLIT:.+]]:
; CHECK: br label %[[FORCONDPREHEADER:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND26PREHEADER:.+]]:
; CHECK: br label %[[FORCOND26:.+]]

; CHECK: [[IFEND]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[GLOOPEXIT2:.+]]

; CHECK: [[WHILEBODYUNIFORM]]:
; CHECK: %[[CMPUNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMPUNIFORM]], label %[[IFTHENUNIFORM:.+]], label %[[IFENDUNIFORM:.+]]

; CHECK: [[IFENDUNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[GLOOPEXIT1UNIFORM:.+]], label %[[IFENDUNIFORMBOSCCINDIR:.+]]

; CHECK: [[GLOOPEXIT1UNIFORM]]:
; CHECK: br label %[[GUNIFORM:.+]]

; CHECK: [[IFENDUNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODYUNIFORM]], label %[[IFENDUNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFENDUNIFORMBOSCCSTORE]]:
; CHECK: br label %[[WHILEBODY]]

; CHECK: [[IFTHENUNIFORM]]
; CHECK: br i1 %{{.+}}, label %[[FORCOND26PREHEADERUNIFORM:.+]], label %[[IFTHENUNIFORMBOSCCINDIR:.+]]

; CHECK: [[FORCONDPREHEADERUNIFORM:.+]]:
; CHECK: br label %[[FORCONDUNIFORM:.+]]

; CHECK: [[FORCONDUNIFORM]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODYUNIFORM:.+]], label %[[GLOOPEXITUNIFORM:.+]]

; CHECK: [[FORBODYUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM]]

; CHECK: [[GLOOPEXITUNIFORM]]:
; CHECK: br label %[[GUNIFORM]]

; CHECK: [[FORCOND26PREHEADERUNIFORM]]:
; CHECK: br label %[[FORCOND26UNIFORM:.+]]

; CHECK: [[IFTHENUNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[FORCONDPREHEADERUNIFORM]], label %[[WHILEBODYPUREEXIT]]

; CHECK: [[FORCOND26UNIFORM]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY30UNIFORM:.+]], label %[[I38LOOPEXITUNIFORM:.+]]

; CHECK: [[FORBODY30UNIFORM]]:
; CHECK: br label %[[FORCOND26UNIFORM]]

; CHECK: [[I38LOOPEXITUNIFORM]]:
; CHECK: br label %[[I38UNIFORM:.+]]

; CHECK: [[GUNIFORM]]:
; CHECK: br label %[[FORCOND17UNIFORM:.+]]

; CHECK: [[FORCOND17UNIFORM]]:
; CHECK: %[[CMP18UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP18UNIFORM]], label %[[FORBODY20UNIFORM:.+]], label %[[HUNIFORM:.+]]

; CHECK: [[FORBODY20UNIFORM]]:
; CHECK: br label %[[FORCOND17UNIFORM]]

; CHECK: [[HUNIFORM]]:
; CHECK: %[[CMP35UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP35UNIFORM]], label %[[I38UNIFORM]], label %[[IFELSE40UNIFORM:.+]]

; CHECK: [[IFELSE40UNIFORM]]:
; CHECK: br label %[[IFEND42UNIFORM:.+]]

; CHECK: [[I38UNIFORM]]:
; CHECK: br label %[[IFEND42:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[GLOOPEXIT:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[GLOOPEXIT]]:
; CHECK: br label %[[FORCOND26PREHEADER]]

; CHECK: [[GLOOPEXIT2]]:
; CHECK: br label %[[GLOOPEXIT2ELSE:.+]]

; CHECK: [[GLOOPEXIT2ELSE]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHENELSE]], label %[[IFTHENSPLIT]]

; CHECK: [[G]]:
; CHECK: br label %[[FORCOND17:.+]]

; CHECK: [[FORCOND17]]:
; CHECK: %[[CMP18:.+]] = icmp
; CHECK: br i1 %[[CMP18]], label %[[FORBODY20:.+]], label %[[H:.+]]

; CHECK: [[FORBODY20]]:
; CHECK: br label %[[FORCOND17]]

; CHECK: [[FORCOND26]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY30:.+]], label %[[I38LOOPEXIT:.+]]

; CHECK: [[FORBODY30]]:
; CHECK: br label %[[FORCOND26]]

; CHECK: [[H]]:
; CHECK: %[[CMP35:.+]] = icmp
; CHECK: br i1 %[[CMP35]], label %[[I38:.+]], label %[[IFELSE40:.+]]

; CHECK: [[I38LOOPEXIT]]:
; CHECK: br label %[[G]]

; CHECK: [[I38]]:
; CHECK: br label %[[IFEND42]]

; CHECK: [[IFELSE40]]:
; CHECK: br label %[[I38]]

; CHECK: [[IFEND42]]:
; CHECK: ret void
