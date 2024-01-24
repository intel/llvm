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

; RUN: veczc -k partial_linearization19 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;       a
;       |
;       b <----.
;      / \     |
;     c   \    |
;    / \   \   |
;   d   e   f -'
;   |   |   |
;    \  \   g
;     \  \ / \
;      \  h   i <,
;       \  \ /  /
;        \  j  /
;         \   /
;          `-'
;
; * where nodes b, c, and g are uniform branches, and node f is a varying
;   branch.
; * where nodes g, h, i and j are divergent.
;
; With partial linearization, it can be transformed in the following way:
;
;       a
;       |
;       b <----.
;      / \     |
;     c   \    |
;    / \   \   |
;   d   e   f -'
;   |   |   |
;    \  |  /
;     \ | /
;      \|/
;       g
;       |
;       i
;       |
;       h
;       |
;       j
;
; The uniform branch `g` has been linearized because both its successors are
; divergent. Not linearizing `g`  would mean that only one of both
; successors could be executed in addition to the other, pending a uniform
; condition evaluates to true, whereas what we want is to possibly execute both
; no matter what the uniform condition evaluates to.
;
; __kernel void partial_linearization19(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;   int i = 0;
;
;   while (1) {
;     if (n > 5) {
;       if (n == 6) {
;         goto d;
;       } else {
;         goto e;
;       }
;     }
;     if (++i + id > 3) {
;       break;
;     }
;   }
;
;   // g
;   if (n == 3) {
;     goto h;
;   } else {
;     goto i;
;   }
;
; d:
;   for (int i = 0; i < n + 5; i++) ret += 2;
;   goto i;
;
; e:
;   for (int i = 1; i < n * 2; i++) ret += i;
;   goto h;
;
; i:
;   for (int i = 0; i < n + 5; i++) ret++;
;   goto j;
;
; h:
;   for (int i = 0; i < n; i++) ret++;
;   goto j;
;
; j:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization19(i32 addrspace(1)* %out, i32 noundef %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %cmp = icmp sgt i32 %n, 5
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  %cmp2 = icmp eq i32 %n, 6
  br i1 %cmp2, label %for.cond, label %for.cond20

if.end:                                           ; preds = %while.body
  %inc = add nuw nsw i32 %i.0, 1
  %add = add nsw i32 %inc, %conv
  %cmp5 = icmp sgt i32 %add, 3
  br i1 %cmp5, label %while.end, label %while.body

while.end:                                        ; preds = %if.end
  %cmp9 = icmp eq i32 %n, 3
  br i1 %cmp9, label %h, label %i28

for.cond:                                         ; preds = %for.body, %if.then
  %ret.0 = phi i32 [ %add17, %for.body ], [ 0, %if.then ]
  %storemerge3 = phi i32 [ %inc18, %for.body ], [ 0, %if.then ]
  %add14 = add nsw i32 %n, 5
  %cmp15 = icmp slt i32 %storemerge3, %add14
  br i1 %cmp15, label %for.body, label %i28

for.body:                                         ; preds = %for.cond
  %add17 = add nuw nsw i32 %ret.0, 2
  %inc18 = add nuw nsw i32 %storemerge3, 1
  br label %for.cond

for.cond20:                                       ; preds = %for.body23, %if.then
  %ret.1 = phi i32 [ %add24, %for.body23 ], [ 0, %if.then ]
  %storemerge2 = phi i32 [ %inc26, %for.body23 ], [ 1, %if.then ]
  %mul = shl nsw i32 %n, 1
  %cmp21 = icmp slt i32 %storemerge2, %mul
  br i1 %cmp21, label %for.body23, label %h

for.body23:                                       ; preds = %for.cond20
  %add24 = add nuw nsw i32 %storemerge2, %ret.1
  %inc26 = add nuw nsw i32 %storemerge2, 1
  br label %for.cond20

i28:                                              ; preds = %for.cond, %while.end
  %ret.2 = phi i32 [ 0, %while.end ], [ %ret.0, %for.cond ]
  br label %for.cond30

for.cond30:                                       ; preds = %for.body34, %i28
  %ret.3 = phi i32 [ %ret.2, %i28 ], [ %inc35, %for.body34 ]
  %storemerge = phi i32 [ 0, %i28 ], [ %inc37, %for.body34 ]
  %add31 = add nsw i32 %n, 5
  %cmp32 = icmp slt i32 %storemerge, %add31
  br i1 %cmp32, label %for.body34, label %j

for.body34:                                       ; preds = %for.cond30
  %inc35 = add nuw nsw i32 %ret.3, 1
  %inc37 = add nuw nsw i32 %storemerge, 1
  br label %for.cond30

h:                                                ; preds = %for.cond20, %while.end
  %ret.4 = phi i32 [ 0, %while.end ], [ %ret.1, %for.cond20 ]
  br label %for.cond40

for.cond40:                                       ; preds = %for.body43, %h
  %ret.5 = phi i32 [ %ret.4, %h ], [ %inc44, %for.body43 ]
  %storemerge1 = phi i32 [ 0, %h ], [ %inc46, %for.body43 ]
  %cmp41 = icmp slt i32 %storemerge1, %n
  br i1 %cmp41, label %for.body43, label %j

for.body43:                                       ; preds = %for.cond40
  %inc44 = add nsw i32 %ret.5, 1
  %inc46 = add nuw nsw i32 %storemerge1, 1
  br label %for.cond40

j:                                                ; preds = %for.cond40, %for.cond30
  %ret.6 = phi i32 [ %ret.3, %for.cond30 ], [ %ret.5, %for.cond40 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.6, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization19, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization19
; CHECK: br label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[IFTHEN:.+]], label %[[IFEND:.+]]

; CHECK: [[IFTHEN]]:
; CHECK: %[[CMP2:.+]] = icmp
; CHECK: br label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[IFTHENELSE:.+]]:
; CHECK: br label %[[H:.+]]

; CHECK: [[IFTHENSPLIT:.+]]:
; CHECK: br i1 %[[CMP2MERGE:.+]], label %[[FORCONDPREHEADER:.+]], label %[[FORCOND20PREHEADER:.+]]

; CHECK: [[FORCOND20PREHEADER]]:
; CHECK: br label %[[FORCOND20:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[IFEND]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: %[[CMP2MERGE]] = phi i1 [ %[[CMP2]], %[[IFTHEN]] ], [ false, %[[IFEND]] ]
; CHECK: br label %[[WHILEEND:.+]]

; CHECK: [[WHILEEND]]:
; CHECK: br label %[[WHILEENDELSE:.+]]

; CHECK: [[WHILEENDELSE]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHENELSE]], label %[[IFTHENSPLIT]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[I28LOOPEXIT:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[FORCOND20]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY23:.+]], label %[[HLOOPEXIT:.+]]

; CHECK: [[FORBODY23]]:
; CHECK: br label %[[FORCOND20]]

; CHECK: [[I28LOOPEXIT]]:
; CHECK: br label %[[H:.+]]

; CHECK: [[I28:.+]]:
; CHECK: br label %[[FORCOND30:.+]]

; CHECK: [[FORCOND30]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY34:.+]], label %[[JLOOPEXIT:.+]]

; CHECK: [[FORBODY34]]:
; CHECK: br label %[[FORCOND30]]

; CHECK: [[HLOOPEXIT]]:
; CHECK: br label %[[H]]

; CHECK: [[H]]:
; CHECK: br label %[[FORCOND40:.+]]

; CHECK: [[FORCOND40]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY43:.+]], label %[[JLOOPEXIT2:.+]]

; CHECK: [[FORBODY43]]:
; CHECK: br label %[[FORCOND40]]

; CHECK: [[JLOOPEXIT]]:
; CHECK: br label %[[J:.+]]

; CHECK: [[JLOOPEXIT2]]:
; CHECK: br label %[[I28]]

; CHECK: [[J]]:
; CHECK: ret void
