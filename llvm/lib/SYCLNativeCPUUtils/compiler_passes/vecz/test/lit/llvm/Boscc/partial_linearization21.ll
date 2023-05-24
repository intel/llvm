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

; RUN: %veczc -k partial_linearization21 -vecz-passes=vecz-loop-rotate,cfg-convert -vecz-choices=LinearizeBOSCC -S < %s | %filecheck %s

; The CFG of the following kernel is:
;
;     a
;     |
;     b <------.
;    / \       |
;   |   c <--. |
;   |  / \   | |
;   | |   d -' |
;   | |  / \   |
;   | | |   e -'
;   | | |  /
;   | | | /
;   | | |/
;   | | /
;    \|/
;     f
;
; * where nodes b, d, and e are uniform branches, and node c is a varying
;   branch.
; * where nodes b, d, e and f are divergent.
;
; With BOSCC, it will be transformed as follows:
;
;     a
;     |
;     b <------.   b' <--.
;    / \       |   |     |
;   |   c <--. |   c' <. |
;   |  / \___|_|__ |   | |
;   | |   d -' |  `d' -' |
;   | |  / \   |   |     |
;   | | |   e -'   e' ---'
;   | | |  /       |
;   | | | /        f'
;   | | |/         |
;   | | /          |
;    \|/          /
;     f --> & <--'
;
; where '&' represents merge blocks of BOSCC regions.
;
; __kernel void partial_linearization21(__global int *out, int n) {
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
;         goto f;
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
;       goto f;
;     }
;   }
;
; f:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization21(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %e, %entry
  %n.off = add i32 %n, -1
  %0 = icmp ult i32 %n.off, 4
  %cmp6 = icmp slt i32 %n, 3
  %or.cond1 = or i1 %cmp6, %0
  br i1 %or.cond1, label %f, label %if.else

while.body5:                                      ; preds = %d
  %cmp6.old = icmp eq i32 %n, 3
  br i1 %cmp6.old, label %if.else, label %f

if.else:                                          ; preds = %while.body5, %while.body
  %cmp9 = icmp sge i32 %conv, %n
  %and = and i32 %n, 1
  %tobool = icmp eq i32 %and, 0
  %or.cond2 = or i1 %tobool, %cmp9
  br i1 %or.cond2, label %d, label %f

d:                                                ; preds = %if.else
  %cmp16 = icmp sgt i32 %n, 3
  br i1 %cmp16, label %e, label %while.body5

e:                                                ; preds = %d
  %and20 = and i32 %n, 1
  %tobool21 = icmp eq i32 %and20, 0
  br i1 %tobool21, label %while.body, label %f

f:                                                ; preds = %e, %if.else, %while.body5, %while.body
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 0, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization21, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization21
; CHECK: br i1 true, label %[[WHILEBODYUNIFORM:.+]], label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: br label %[[IFELSEPREHEADER:.+]]

; CHECK: [[IFELSEPREHEADER]]:
; CHECK: br label %[[IFELSE:.+]]

; CHECK: [[WHILEBODY5:.+]]:

; CHECK: br i1 %{{.+}}, label %[[IFELSE]], label %[[IFELSEPUREEXIT:.+]]

; CHECK: [[IFELSEPUREEXIT]]:
; CHECK: br label %[[E:.+]]

; CHECK: [[IFELSE]]:
; CHECK: br label %[[D:.+]]

; CHECK: [[WHILEBODYUNIFORM]]:
; CHECK: %[[CMP6UNIFORM:cmp.+]] = icmp
; CHECK: %[[ORCOND1UNIFORM:.+]] = or i1 %[[CMP6UNIFORM]]
; CHECK: br i1 %[[ORCOND1UNIFORM]], label %[[FLOOPEXIT1UNIFORM:.+]], label %[[IFELSEPREHEADERUNIFORM:.+]]

; CHECK: [[IFELSEPREHEADERUNIFORM]]:
; CHECK: br label %[[IFELSEUNIFORM:.+]]

; CHECK: [[IFELSEUNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[DUNIFORM:.+]], label %[[IFELSEUNIFORMBOSCCINDIR:.+]]

; CHECK: [[DUNIFORM]]:
; CHECK: %[[CMP16UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP16UNIFORM]], label %[[EUNIFORM:.+]], label %[[WHILEBODY5UNIFORM:.+]]

; CHECK: [[IFELSEUNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[FLOOPEXITUNIFORM:.+]], label %[[IFELSEUNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFELSEUNIFORMBOSCCSTORE]]:
; CHECK: br label %[[D]]

; CHECK: [[WHILEBODY5UNIFORM]]:
; CHECK: %[[CMP6OLDUNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP6OLDUNIFORM]], label %[[IFELSEUNIFORM]], label %[[FLOOPEXITUNIFORM]]

; CHECK: [[EUNIFORM]]:
; CHECK: %[[TOBOOL21UNIFORM:.+]] = icmp
; CHECK: br i1 %[[TOBOOL21UNIFORM]], label %[[WHILEBODYUNIFORM]], label %[[FLOOPEXIT1UNIFORM]]


; CHECK: [[FLOOPEXITUNIFORM]]:
; CHECK: br label %[[FUNIFORM:.+]]

; CHECK: [[FLOOPEXIT1UNIFORM]]:
; CHECK: br label %[[F:.+]]

; CHECK: [[D]]:
; CHECK: br label %[[WHILEBODY5]]

; CHECK: [[E]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[FLOOPEXIT:.+]]

; CHECK: [[FLOOPEXIT]]:
; CHECK: br label %[[FLOOPEXITELSE:.+]]

; CHECK: [[FLOOPEXITELSE]]:
; CHECK: br label %[[FLOOPEXIT1:.+]]

; CHECK: [[FLOOPEXIT1]]:
; CHECK: br label %[[F]]

; CHECK: [[F]]:
; CHECK: ret void
