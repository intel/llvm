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

; RUN: veczc -k partial_linearization5 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;     a
;    / \
;   b   c
;   |\ / \
;   | d   e
;   |  \ /
;   |   f
;    \ /
;     g
;
; * where node c is a uniform branch, and nodes a and b are varying branches.
; * where nodes b, c, d, f, g are divergent.
;
; With partial linearization we will have a CFG of the form:
;
;     a
;     |
;     c
;    / \
;   |   e
;    \ /
;     b
;     |
;     d
;     |
;     f
;     |
;     g
;
; __kernel void partial_linearization5(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   if (id % 2 == 0) { // a
;     if (id == 4) { // b
;       goto g;
;     } else {
;       goto d;
;     }
;   } else { // c
;     if (n % 2 == 0) {
;       goto d;
;     } else {
;       goto e;
;     }
;   }
;
; d:
;   for (int i = 0; i < n / 4; i++) { ret += i - 2; }
;   goto f;
;
; e:
;   for (int i = 0; i < n + 5; i++) { ret += i + 5; }
;
; f:
;   ret *= ret % n;
;   ret *= ret + 4;
;
; g:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization5(i32 addrspace(1)* %out, i32 noundef %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  %rem1 = and i32 %conv, 1
  %cmp = icmp eq i32 %rem1, 0
  br i1 %cmp, label %if.then, label %if.else5

if.then:                                          ; preds = %entry
  %cmp2 = icmp eq i32 %conv, 4
  br i1 %cmp2, label %g, label %d

if.else5:                                         ; preds = %entry
  %rem62 = and i32 %n, 1
  %cmp7 = icmp eq i32 %rem62, 0
  br i1 %cmp7, label %d, label %e

d:                                                ; preds = %if.else5, %if.then
  br label %for.cond

for.cond:                                         ; preds = %for.body, %d
  %ret.0 = phi i32 [ 0, %d ], [ %add, %for.body ]
  %storemerge3 = phi i32 [ 0, %d ], [ %inc, %for.body ]
  %div = sdiv i32 %n, 4
  %cmp11 = icmp slt i32 %storemerge3, %div
  br i1 %cmp11, label %for.body, label %f

for.body:                                         ; preds = %for.cond
  %sub = add i32 %ret.0, -2
  %add = add i32 %sub, %storemerge3
  %inc = add nsw i32 %storemerge3, 1
  br label %for.cond

e:                                                ; preds = %if.else5
  br label %for.cond14

for.cond14:                                       ; preds = %for.body18, %e
  %ret.1 = phi i32 [ 0, %e ], [ %add20, %for.body18 ]
  %storemerge = phi i32 [ 0, %e ], [ %inc22, %for.body18 ]
  %add15 = add nsw i32 %n, 5
  %cmp16 = icmp slt i32 %storemerge, %add15
  br i1 %cmp16, label %for.body18, label %f

for.body18:                                       ; preds = %for.cond14
  %add19 = add i32 %ret.1, 5
  %add20 = add i32 %add19, %storemerge
  %inc22 = add nsw i32 %storemerge, 1
  br label %for.cond14

f:                                                ; preds = %for.cond14, %for.cond
  %ret.2 = phi i32 [ %ret.0, %for.cond ], [ %ret.1, %for.cond14 ]
  %0 = icmp eq i32 %ret.2, -2147483648
  %1 = icmp eq i32 %n, -1
  %2 = and i1 %1, %0
  %3 = icmp eq i32 %n, 0
  %4 = or i1 %3, %2
  %5 = select i1 %4, i32 1, i32 %n
  %rem24 = srem i32 %ret.2, %5
  %mul = mul nsw i32 %rem24, %ret.2
  %add25 = add nsw i32 %mul, 4
  %mul26 = mul nsw i32 %add25, %mul
  br label %g

g:                                                ; preds = %f, %if.then
  %ret.3 = phi i32 [ %mul26, %f ], [ 0, %if.then ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.3, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization5, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization5
; CHECK: br label %[[IFELSE5:.+]]

; CHECK: [[IFTHEN:.+]]:
; CHECK: br label %[[D:.+]]

; CHECK: [[IFELSE5]]:
; CHECK: %[[CMP7:.+]] = icmp
; CHECK: br i1 %[[CMP7]], label %[[IFTHEN]], label %[[FORCOND14PREHEADER:.+]]

; CHECK: [[FORCOND14PREHEADER]]:
; CHECK: br label %[[FORCOND14:.+]]

; CHECK: [[D]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: %[[CMP11:.+]] = icmp
; CHECK: br i1 %[[CMP11]], label %[[FORBODY:.+]], label %[[FLOOPEXIT:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[FORCOND14]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY18:.+]], label %[[FLOOPEXIT2:.+]]

; CHECK: [[FORBODY18]]:
; CHECK: br label %[[FORCOND14]]

; CHECK: [[FLOOPEXIT]]:
; CHECK: br label %[[F:.+]]

; CHECK: [[FLOOPEXIT2]]:
; CHECK: br label %[[IFTHEN]]

; CHECK: [[F]]:
; CHECK: br label %[[G:.+]]

; CHECK: [[G]]:
; CHECK: ret void
