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

; REQUIRES: !llvm-19+
; RUN: veczc -k partial_linearization22 -vecz-passes="function(lowerswitch),vecz-loop-rotate,indvars,cfg-convert" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;     a
;     |
;     b <------.
;    / \       |
;   f   c <--. |
;   |\ / \   | |
;   | |   d -' |
;   | |\ / \   |
;   | | |   e -'
;   | | |\ /
;   | | | g
;   | | |/
;   | | /
;    \|/
;     h
;
; * where nodes b, d, and e are uniform branches, and node c is a varying
;   branch.
; * where nodes b, d, e and f are divergent.
;
; With partial linearization, it will be transformed as follows:
;
;     a
;     |
;     b <--.
;    /|    |
;   f c <. |
;   | |  | |
;   | d -' |
;   | |    |
;   | e ---'
;    \|
;     g
;     |
;     h
;
; __kernel void partial_linearization22(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   while (1) {
;     if (n > 0 && n < 5) {
;       goto f;
;     }
;     while (1) {
;       if (n <= 2) {
;         goto f;
;       } else {
;         if (ret + id >= n) {
;           goto d;
;         }
;       }
;       if (n & 1) {
;         goto h;
;       }
;
; d:
;       if (n > 3) {
;         goto e;
;       }
;     }
;
; e:
;     if (n & 1) {
;       goto g;
;     }
;   }
;
; f:
;   if (n == 2) {
;     goto h;
;   }
;
; g:
;   for (int i = 0; i < n + 1; i++) ret++;
;   goto h;
;
; h:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization22(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %e, %entry
  %n.off = add i32 %n, -1
  %0 = icmp ult i32 %n.off, 4
  %cmp6 = icmp slt i32 %n, 3
  %or.cond1 = or i1 %cmp6, %0
  br i1 %or.cond1, label %f, label %if.else

while.body5:                                      ; preds = %d
  switch i32 %n, label %g [
    i32 3, label %if.else
    i32 2, label %h
  ]

if.else:                                          ; preds = %while.body5, %while.body
  %cmp9 = icmp sge i32 %conv, %n
  %and = and i32 %n, 1
  %tobool = icmp eq i32 %and, 0
  %or.cond2 = or i1 %tobool, %cmp9
  br i1 %or.cond2, label %d, label %h

d:                                                ; preds = %if.else
  %cmp16 = icmp sgt i32 %n, 3
  br i1 %cmp16, label %e, label %while.body5

e:                                                ; preds = %d
  %and20 = and i32 %n, 1
  %tobool21 = icmp eq i32 %and20, 0
  br i1 %tobool21, label %while.body, label %g

f:                                                ; preds = %while.body
  %cmp24 = icmp eq i32 %n, 2
  br i1 %cmp24, label %h, label %g

g:                                                ; preds = %f, %e, %while.body5
  br label %for.cond

for.cond:                                         ; preds = %for.body, %g
  %ret.0 = phi i32 [ 0, %g ], [ %inc, %for.body ]
  %storemerge = phi i32 [ 0, %g ], [ %inc31, %for.body ]
  %cmp29 = icmp sgt i32 %storemerge, %n
  br i1 %cmp29, label %h, label %for.body

for.body:                                         ; preds = %for.cond
  %inc = add nuw nsw i32 %ret.0, 1
  %inc31 = add nuw nsw i32 %storemerge, 1
  br label %for.cond

h:                                                ; preds = %for.cond, %f, %if.else, %while.body5
  %ret.1 = phi i32 [ 0, %f ], [ %ret.0, %for.cond ], [ 0, %if.else ], [ 0, %while.body5 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.1, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization22, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization22
; CHECK: br label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: %[[CMP6:.+]] = icmp slt
; CHECK: %[[ORCOND1:.+]] = or i1 %[[CMP6]]
; CHECK: %[[F_EXIT_MASK:.+]] = select i1
; CHECK: %[[ORCOND2:.+]] = call i1 @__vecz_b_divergence_any(i1 %[[ORCOND1]])
; CHECK: br i1 %[[ORCOND2]], label %[[F:.+]], label %[[IFELSEPREHEADER:.+]]

; CHECK: [[IFELSEPREHEADER]]:
; CHECK: br label %[[IFELSE:.+]]

; CHECK: [[LEAFBLOCK1:.*]]:
; CHECK: %[[SWITCHLEAF:.+]] = icmp eq i32 %n, 3
; CHECK: br i1 %{{.+}}, label %[[IFELSE]], label %[[IFELSEPUREEXIT:.+]]

; CHECK: [[IFELSEPUREEXIT]]:
; CHECK: br label %[[E:.+]]

; CHECK: [[IFELSE]]:
; CHECK: br label %[[D:.+]]

; CHECK: [[D]]:
; CHECK: br label %[[LEAFBLOCK1]]

; CHECK: [[E]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: %[[CMP24MERGE:.+]] = phi i1 [ %[[G_EXIT_MASK:.+]], %[[F]] ], [ false, %[[E]] ]
; CHECK: br label %[[HLOOPEXIT1:.+]]

; CHECK: [[F]]:
; CHECK: %[[CMP24:.+]] = icmp eq i32 %n, 2
; CHECK: %[[G_EXIT_MASK]] = select i1 %[[CMP24]], i1 false, i1 %[[F_EXIT_MASK]]
; CHECK: br label %[[WHILEBODYPUREEXIT]]

; CHECK: [[FELSE:.+]]:
; CHECK: br label %[[G:.+]]

; CHECK: [[FSPLIT:.+]]:
; CHECK: %[[CMP24_ANY:.+]] = call i1 @__vecz_b_divergence_any(i1 %cmp24.merge)
; CHECK: br i1 %[[CMP24_ANY]], label %[[H:.+]], label %[[G]]

; CHECK: [[GLOOPEXIT:.+]]:
; CHECK: br label %[[GLOOPEXITELSE:.+]]

; CHECK: [[GLOOPEXITELSE]]:
; CHECK: br i1 %{{.+}}, label %[[FELSE]], label %[[FSPLIT]]

; CHECK: [[G]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 true, label %[[HLOOPEXIT:.+]], label %[[FORBODY:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]



; CHECK: [[HLOOPEXIT]]:
; CHECK: br label %[[H:.+]]

; CHECK: [[HLOOPEXIT1]]:
; CHECK: br label %[[HLOOPEXIT1ELSE:.+]]

; CHECK: [[HLOOPEXIT1ELSE]]:
; CHECK: br label %[[GLOOPEXIT]]

;; CHECK: [[H]]:
;; CHECK: ret void
