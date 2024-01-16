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

; RUN: veczc -k partial_linearization2 -vecz-passes="function(instcombine,simplifycfg),mergereturn,vecz-loop-rotate,function(loop(indvars)),cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;         a
;        / \
;       /   \
;      /     \
;     b       c
;    / \     / \
;   d   e   f   g
;    \   \ /   /
;     \   X   /
;      \ / \ /
;       h   i
;        \ /
;         j
;
; * where node a is a uniform branch, and nodes b and c are varying branches.
; * where nodes d, e, f, g are divergent.
;
; With partial linearization we will have a CFG of the form:
;
;         a
;        / \
;       /   \
;      /     \
;     b       c
;    /         \
;   e - d   f - g
;        \ /
;         i
;         |
;         h
;         |
;         j
;
; __kernel void partial_linearization2(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   if (n < 10) { // uniform
;     if (id % 3 == 0) { // varying
;       for (int i = 0; i < n - 1; i++) { ret /= 2; } goto h;
;     } else { // varying
;       for (int i = 0; i < n / 3; i++) { ret -= 2; } goto i;
;     }
;   } else { // uniform
;     if (id % 2 == 0) { // varying
;       for (int i = 0; i < n * 2; i++) { ret += 1; } goto h;
;     } else { // varying
;       for (int i = 0; i < n + 5; i++) { ret *= 2; } goto i;
;     }
;   }
;
; h:
;   ret += 5;
;   goto end;
;
; i:
;   ret *= 10;
;   goto end;
;
; end:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization2(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  %cmp = icmp slt i32 %n, 10
  br i1 %cmp, label %if.then, label %if.else17

if.then:                                          ; preds = %entry
  %rem = srem i32 %conv, 3
  %cmp2 = icmp eq i32 %rem, 0
  br i1 %cmp2, label %if.then4, label %if.else

if.then4:                                         ; preds = %if.then
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.then4
  %ret.0 = phi i32 [ 0, %if.then4 ], [ %div, %for.body ]
  %storemerge5 = phi i32 [ 0, %if.then4 ], [ %inc, %for.body ]
  %sub = add nsw i32 %n, -1
  %cmp5 = icmp slt i32 %storemerge5, %sub
  br i1 %cmp5, label %for.body, label %h

for.body:                                         ; preds = %for.cond
  %div = sdiv i32 %ret.0, 2
  %inc = add nsw i32 %storemerge5, 1
  br label %for.cond

if.else:                                          ; preds = %if.then
  br label %for.cond8

for.cond8:                                        ; preds = %for.body12, %if.else
  %ret.1 = phi i32 [ 0, %if.else ], [ %sub13, %for.body12 ]
  %storemerge4 = phi i32 [ 0, %if.else ], [ %inc15, %for.body12 ]
  %div9 = sdiv i32 %n, 3
  %cmp10 = icmp slt i32 %storemerge4, %div9
  br i1 %cmp10, label %for.body12, label %i42

for.body12:                                       ; preds = %for.cond8
  %sub13 = add nsw i32 %ret.1, -2
  %inc15 = add nsw i32 %storemerge4, 1
  br label %for.cond8

if.else17:                                        ; preds = %entry
  %rem181 = and i32 %conv, 1
  %cmp19 = icmp eq i32 %rem181, 0
  br i1 %cmp19, label %if.then21, label %if.else30

if.then21:                                        ; preds = %if.else17
  br label %for.cond23

for.cond23:                                       ; preds = %for.body26, %if.then21
  %ret.2 = phi i32 [ 0, %if.then21 ], [ %add, %for.body26 ]
  %storemerge3 = phi i32 [ 0, %if.then21 ], [ %inc28, %for.body26 ]
  %mul = shl nsw i32 %n, 1
  %cmp24 = icmp slt i32 %storemerge3, %mul
  br i1 %cmp24, label %for.body26, label %h

for.body26:                                       ; preds = %for.cond23
  %add = add nsw i32 %ret.2, 1
  %inc28 = add nsw i32 %storemerge3, 1
  br label %for.cond23

if.else30:                                        ; preds = %if.else17
  br label %for.cond32

for.cond32:                                       ; preds = %for.body36, %if.else30
  %ret.3 = phi i32 [ 0, %if.else30 ], [ %mul37, %for.body36 ]
  %storemerge = phi i32 [ 0, %if.else30 ], [ %inc39, %for.body36 ]
  %add33 = add nsw i32 %n, 5
  %cmp34 = icmp slt i32 %storemerge, %add33
  br i1 %cmp34, label %for.body36, label %i42

for.body36:                                       ; preds = %for.cond32
  %mul37 = shl nsw i32 %ret.3, 1
  %inc39 = add nsw i32 %storemerge, 1
  br label %for.cond32

h:                                                ; preds = %for.cond23, %for.cond
  %ret.4 = phi i32 [ %ret.0, %for.cond ], [ %ret.2, %for.cond23 ]
  %add41 = add nsw i32 %ret.4, 5
  br label %end

i42:                                              ; preds = %for.cond32, %for.cond8
  %ret.5 = phi i32 [ %ret.1, %for.cond8 ], [ %ret.3, %for.cond32 ]
  %mul43 = mul nsw i32 %ret.5, 10
  br label %end

end:                                              ; preds = %i42, %h
  %storemerge2 = phi i32 [ %mul43, %i42 ], [ %add41, %h ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %storemerge2, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization2, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization2
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[IFTHEN:.+]], label %[[IFELSE17:.+]]

; CHECK: [[IFTHEN]]:
; CHECK: br label %[[FORCOND8PREHEADER:.+]]

; CHECK: [[FORCOND8PREHEADER:.+]]:
; CHECK: br label %[[FORCOND8:.+]]

; CHECK: [[FORCONDPREHEADER:.+]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: %[[CMP5:.+]] = icmp
; CHECK: br i1 %[[CMP5]], label %[[FORBODY:.+]], label %[[HLOOPEXIT:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[FORCOND8]]:
; CHECK: %[[CMP10:.+]] = icmp
; CHECK: br i1 %[[CMP10]], label %[[FORBODY12:.+]], label %[[I42LOOPEXIT:.+]]

; CHECK: [[FORBODY12]]:
; CHECK: br label %[[FORCOND8]]

; CHECK: [[IFELSE17]]:
; CHECK: br label %[[FORCOND32PREHEADER:.+]]

; CHECK: [[FORCOND32PREHEADER]]:
; CHECK: br label %[[FORCOND32:.+]]

; CHECK: [[FORCOND23PREHEADER:.+]]:
; CHECK: br label %[[FORCOND23:.+]]

; CHECK: [[FORCOND23]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY26:.+]], label %[[HLOOPEXIT3:.+]]

; CHECK: [[FORBODY26]]:
; CHECK: br label %[[FORCOND23]]

; CHECK: [[FORCOND32]]:
; CHECK: br i1 false, label %[[FORBODY36:.+]], label %[[I42LOOPEXIT4:.+]]

; CHECK: [[FORBODY36]]:
; CHECK: br label %[[FORCOND32]]

; CHECK: [[HLOOPEXIT]]:
; CHECK: br label %[[I42:.+]]

; CHECK: [[HLOOPEXIT3]]:
; CHECK: br label %[[I42]]

; CHECK: [[H:.+]]:
; CHECK: br label %[[END:.+]]

; CHECK: [[I42LOOPEXIT]]:
; CHECK: br label %[[FORCONDPREHEADER]]

; CHECK: [[I42LOOPEXIT4]]:
; CHECK: br label %[[FORCOND23PREHEADER]]

; CHECK: [[I42]]:
; CHECK: br label %[[H]]

; CHECK: [[END]]:
; CHECK: ret void
