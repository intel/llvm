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

; RUN: veczc -k partial_linearization6 -vecz-passes="function(simplifycfg),vecz-loop-rotate,cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;       a
;       |
;       b <-.
;      / \  |
;     c   d |
;    / \ /  |
;   e   f --'
;    \  |
;     \ g
;      \|
;       h
;
; * where nodes b and c are uniform branches, and node f is a varying
;   branch.
; * where nodes g and h are divergent.
;
; With BOSCC, it will be transformed as follows:
;
;       a
;       |
;       b <-. .---> b' <-.
;      / \  | |    / \   |
;     c   d | |   c'  d' |
;    / \ /  | |  / \ /   |
;   e   f --' | e'  f' --'
;    \  |\____'  \  |
;     \ g         \ |
;      \|          \|
;       h           g'
;       |           |
;       `---> & <-- h'
;
; where '&' represents merge blocks of BOSCC regions.
;
; __kernel void partial_linearization6(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   while (1) {
;     if (n % 2 == 0) {
;       if (n > 2) {
;         goto e;
;       }
;     } else {
;       ret += n + 1;
;     }
;     if (id == n) break;
;   }
;
;   ret += n * 2;
;   ret /= n;
;   goto early;
;
; e:
;   ret += n * 4;
;   ret -= n;
;
; early:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization6(i32 addrspace(1)* %out, i32 noundef %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end10, %entry
  %ret.0 = phi i32 [ 0, %entry ], [ %ret.1, %if.end10 ]
  %rem1 = and i32 %n, 1
  %cmp = icmp eq i32 %rem1, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  %cmp2 = icmp sgt i32 %n, 2
  br i1 %cmp2, label %e, label %if.end6

if.else:                                          ; preds = %while.body
  %add = add nsw i32 %n, 1
  %add5 = add nsw i32 %add, %ret.0
  br label %if.end6

if.end6:                                          ; preds = %if.else, %if.then
  %ret.1 = phi i32 [ %add5, %if.else ], [ %ret.0, %if.then ]
  %cmp7 = icmp eq i32 %conv, %n
  br i1 %cmp7, label %while.end, label %if.end10

if.end10:                                         ; preds = %if.end6
  br label %while.body

while.end:                                        ; preds = %if.end6
  %mul = shl nsw i32 %n, 1
  %add11 = add nsw i32 %ret.1, %mul
  %0 = icmp eq i32 %add11, -2147483648
  %1 = icmp eq i32 %n, -1
  %2 = and i1 %1, %0
  %3 = icmp eq i32 %n, 0
  %4 = or i1 %3, %2
  %5 = select i1 %4, i32 1, i32 %n
  %div = sdiv i32 %add11, %5
  br label %early

e:                                                ; preds = %if.then
  %mul12 = mul i32 %n, 4
  %n.neg = sub i32 0, %n
  %add13 = add i32 %mul12, %n.neg
  %sub = add i32 %add13, %ret.0
  br label %early

early:                                            ; preds = %e, %while.end
  %storemerge = phi i32 [ %div, %while.end ], [ %sub, %e ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %storemerge, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization6, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization6
; CHECK: br i1 true, label %[[WHILEBODYUNIFORM:.+]], label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[IFTHEN:.+]], label %[[IFELSE:.+]]

; CHECK: [[IFTHEN]]:
; CHECK: %[[CMP2:.+]] = icmp
; CHECK: br i1 %[[CMP2]], label %[[E:.+]], label %[[IFEND6:.+]]

; CHECK: [[IFELSE]]:
; CHECK: br label %[[IFEND6]]

; CHECK: [[IFEND6]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[WHILEEND:.+]]

; CHECK: [[WHILEBODYUNIFORM]]:
; CHECK: %[[CMPUNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMPUNIFORM]], label %[[IFTHENUNIFORM:.+]], label %[[IFELSEUNIFORM:.+]]

; CHECK: [[IFELSEUNIFORM]]:
; CHECK: br label %[[IFEND6UNIFORM:.+]]

; CHECK: [[IFTHENUNIFORM]]:
; CHECK: %[[CMP2UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP2UNIFORM]], label %[[EUNIFORM:.+]], label %[[IFEND6EUNIFORM:.+]]

; CHECK: [[IFEND6UNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEENDUNIFORM:.+]], label %[[IFEND6UNIFORMBOSCCINDIR:.+]]

; CHECK: [[WHILEENDUNIFORM]]:
; CHECK: br label %[[EARLYUNIFORM:.+]]

; CHECK: [[IFEND6UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODYUNIFORM]], label %[[IFEND6UNIFORMBOSCCSTORE:.+]]

; CHECK: [[IFEND6UNIFORMBOSCCSTORE]]:
; CHECK: br label %[[WHILEBODY]]

; CHECK: [[EUNIFORM]]:
; CHECK: br label %[[EARLY:.+]]

; CHECK: [[WHILEEND]]:
; CHECK: br label %[[WHILEENDELSE:.+]]

; CHECK: [[WHILEENDELSE]]:
; CHECK: br i1 %{{.+}}, label %[[EELSE:.+]], label %[[ESPLIT:.+]]

; CHECK: [[E]]:
; CHECK: br label %[[WHILEBODYPUREEXIT]]

; CHECK: [[EELSE]]:
; CHECK: br label %[[EARLY]]

; CHECK: [[ESPLIT]]:
; CHECK: br label %[[EARLY]]

; CHECK: [[EARLY]]:
; CHECK: ret void
