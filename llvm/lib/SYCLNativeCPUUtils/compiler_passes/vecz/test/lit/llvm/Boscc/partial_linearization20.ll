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

; RUN: veczc -k partial_linearization20 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;     a
;     |
;     b <--------.
;    / \         |
;   |   c        |
;   |  / \       |
;   | f   h <--. |
;   | |  / \   | |
;   | | |   d -' |
;   | | |   |    |
;   | | |   e ---'
;   | | |  /
;   | | | /
;   | | |/
;   | | /
;    \|/
;     g
;
; * where nodes b, d, and e are uniform branches, and node h is a varying
;   branch.
; * where nodes b, d and g are divergent.
;
; With BOSCC, it will be transformed as follows:
;
;     a
;     |
;     b <--------.      b' <--.
;    / \         |      |     |
;   |   c        | .-.  c'    |
;   |  / \       | |  \/|     |
;   | f   h <--. | |  / h' <. |
;   | |  / \   | | | f' |   | |
;   | | |   d -' | | |  d' -' |
;   | | |   |\___|_' |  |     |
;   | | |   e ---'   |  e' ---'
;   | | |  /          \ |
;   | | | /            \|
;   | | |/              g'
;   | | /               |
;    \|/               /
;     g ----> & <-----'
;
; where '&' represents merge blocks of BOSCC regions.
;
; __kernel void partial_linearization20(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   while (1) {
;     if (n > 0 && n < 5) {
;       goto g;
;     }
;     if (n == 6) {
;       goto f;
;     }
;     while (1) {
;       if (ret++ + id >= n) {
;         goto d;
;       }
;       if (n & 1) {
;         goto g;
;       }
;
; d:
;       if (n > 3) {
;         goto e;
;       }
;     }
; e:
;     if (n & 1) {
;       goto g;
;     }
;   }
;
; f:
;   for (int i = 0; i < n + 1; i++) ret++;
; g:
;   out[id] = ret;
; }

; ModuleID = 'kernel.opencl'
source_filename = "kernel.opencl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization20(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %e, %entry
  %ret.0 = phi i32 [ 0, %entry ], [ %inc, %e ]
  %n.off = add i32 %n, -1
  %0 = icmp ult i32 %n.off, 4
  br i1 %0, label %g, label %if.end

if.end:                                           ; preds = %while.body
  %cmp4 = icmp eq i32 %n, 6
  br i1 %cmp4, label %for.cond, label %while.body9

while.body9:                                      ; preds = %d, %if.end
  %ret.1 = phi i32 [ %ret.0, %if.end ], [ %inc, %d ]
  %inc = add nsw i32 %ret.1, 1
  %add = add nsw i32 %ret.1, %conv
  %cmp10 = icmp sge i32 %add, %n
  %and = and i32 %n, 1
  %tobool = icmp eq i32 %and, 0
  %or.cond1 = or i1 %tobool, %cmp10
  br i1 %or.cond1, label %d, label %g

d:                                                ; preds = %while.body9
  %cmp16 = icmp sgt i32 %n, 3
  br i1 %cmp16, label %e, label %while.body9

e:                                                ; preds = %d
  %and20 = and i32 %n, 1
  %tobool21 = icmp eq i32 %and20, 0
  br i1 %tobool21, label %while.body, label %g

for.cond:                                         ; preds = %for.body, %if.end
  %ret.2 = phi i32 [ %inc27, %for.body ], [ %ret.0, %if.end ]
  %storemerge = phi i32 [ %inc28, %for.body ], [ 0, %if.end ]
  %cmp25 = icmp sgt i32 %storemerge, %n
  br i1 %cmp25, label %g, label %for.body

for.body:                                         ; preds = %for.cond
  %inc27 = add nsw i32 %ret.2, 1
  %inc28 = add nuw nsw i32 %storemerge, 1
  br label %for.cond

g:                                                ; preds = %for.cond, %e, %while.body9, %while.body
  %ret.3 = phi i32 [ %ret.0, %while.body ], [ %inc, %e ], [ %ret.2, %for.cond ], [ %inc, %while.body9 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.3, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization20, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization20
; CHECK: br i1 true, label %[[WHILEBODYUNIFORM:.+]], label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: br label %[[IFEND:.+]]

; CHECK: [[IFEND]]:
; CHECK: %[[CMP4:.+]] = icmp
; CHECK: br i1 %[[CMP4]], label %[[FORCONDPREHEADER:.+]], label %[[WHILEBODY9PREHEADER:.+]]

; CHECK: [[WHILEBODY9PREHEADER]]:
; CHECK: br label %[[WHILEBODY9:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[FORCONDPREHEADERELSE:.+]]:
; CHECK: br label %[[G:.+]]

; CHECK: [[FORCONDPREHEADERSPLIT:.+]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[WHILEBODY9]]:
; CHECK: br label %[[D:.+]]

; CHECK: [[WHILEBODYUNIFORM:.+]]:
; CHECK: br i1 %{{.+}}, label %[[GLOOPEXIT2UNIFORM:.+]], label %[[IFENDUNIFORM:.+]]

; CHECK: [[IFENDUNIFORM]]:
; CHECK: %[[CMP4UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP4UNIFORM]], label %[[FORCONDPREHEADERUNIFORM:.+]], label %[[WHILEBODY9PREHEADERUNIFORM:.+]]

; CHECK: [[WHILEBODY9PREHEADERUNIFORM]]:
; CHECK: br label %[[WHILEBODY8UNIFORM:.+]]

; CHECK: [[WHILEBODY9UNIFORM:.+]]:
; CHECK: br i1 %{{.+}}, label %[[DUNIFORM:.+]], label %[[WHILEBODY9UNIFORMBOSCCINDIR:.+]]

; CHECK: [[DUNIFORM]]:
; CHECK: %[[CMP16UNIFORM:.+]] = icmp
; CHECK: br i1 %{{.+}}, label %[[EUNIFORM:.+]], label %[[WHILEBODY9UNIFORM]]

; CHECK: [[WHILEBODY9UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[GLOOPEXIT1UNIFORM:.+]], label %[[WHILEBODY9UNIFORMBOSCCSTORE:.+]]

; CHECK: [[WHILEBODY9UNIFORMBOSCCSTORE]]:
; CHECK: br label %[[D]]

; CHECK: [[EUNIFORM]]:
; CHECK: %[[TOBOOL21UNIFORM:.+]] = icmp
; CHECK: br i1 %[[TOBOOL21UNIFORM]], label %[[WHILEBODYUNIFORM]], label %[[GLOOPEXIT2UNIFORM]]


; CHECK: [[GLOOPEXIT1UNIFORM]]:
; CHECK: br label %[[GUNIFORM:.+]]

; CHECK: [[FORCONDPREHEADERUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM:.+]]

; CHECK: [[FORCONDUNIFORM]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(true)}}, label %[[GLOOPEXITUNIFORM:.+]], label %[[FORBODYUNIFORM:.+]]

; CHECK: [[FORBODYUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM]]

; CHECK: [[GLOOPEXITUNIFORM]]:
; CHECK: br label %[[GUNIFORM]]

; CHECK: [[GLOOPEXIT2UNIFORM]]:
; CHECK: br label %[[G]]

; CHECK: [[D]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY9]], label %[[WHILEBODY9PUREEXIT:.+]]

; CHECK: [[WHILEBODY9PUREEXIT]]:
; CHECK: br label %[[E:.+]]

; CHECK: [[E]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[GLOOPEXIT1:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(true)}}, label %[[GLOOPEXIT:.+]], label %[[FORBODY:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[GLOOPEXIT]]:
; CHECK: br label %[[G]]

; CHECK: [[GLOOPEXIT1]]:
; CHECK: br label %[[GLOOPEXIT1ELSE:.+]]

; CHECK: [[GLOOPEXIT1ELSE]]:
; CHECK: br label %[[GLOOPEXIT2:.+]]

; CHECK: [[GLOOPEXIT2]]:
; CHECK: br label %[[GLOOPEXIT2ELSE:.+]]

; CHECK: [[GLOOPEXIT2ELSE]]:
; CHECK: br i1 %{{.+}}, label %[[FORCONDPREHEADERELSE]], label %[[FORCONDPREHEADERSPLIT]]

; CHECK: [[G]]:
; CHECK: ret void
