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

; RUN: veczc -k partial_linearization17 -vecz-passes="function(instcombine,simplifycfg),mergereturn,vecz-loop-rotate,function(loop(indvars)),cfg-convert,cleanup-divergence" -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;              a
;              |
;              b <----.
;             / \     |
;            c   d    |
;           /   / \   |
;          e   f   g -'
;         / \  |   |
;   .--> h   | i   j
;   |   / \  |  \ /
;   '- k   l '-> m
;      |    \   /
;      n     \ /
;       \     o
;        \   /
;         \ /
;          p
;
; * where nodes b, d, and h are uniform branches, and nodes e and g are varying
;   branches.
; * where nodes h, j, m, o, and p are divergent.
;
; With partial linearization, it can be transformed in the following way:
;
;              a
;              |
;              b <----.
;             / \     |
;            c   d    |
;           /   / \   |
;          e   f   g -'
;         /    |   |
;   .--> h     i   |
;   |   / \    |   |
;   '- k   l   |   |
;       \   \  |  /
;        n   \ | /
;         \   \|/
;          `-> j
;              |
;              m
;              |
;              o
;              |
;              p
;
; __kernel void partial_linearization17(__global int *out, int n, int x) {
;   int id = get_global_id(0);
;   int ret = 0;
;   int i = 0;
;
;   while (1) {
;     if (n > 10) {
;       goto c;
;     } else if (n < 5) {
;       goto f;
;     }
;     if (id + i++ % 2 == 0) {
;       break;
;     }
;   }
;
;   // j
;   for (int i = 0; i < n + 10; i++) ret++;
;   goto m;
;
; f:
;   ret += x / 2;
;   for (int i = 0; i < x / 2; i++) ret += i;
;   goto m;
;
; c:
;   for (int i = 0; i < n - 5; i++) ret += 2;
;   // e
;   if (id % 2 == 0) {
;     goto h;
;   } else {
;     goto m;
;   }
;
; m:
;   ret <<= 2;
;   goto o;
;
; h:
;   for (int i = 0; i < x / 2; i++) {
;     if (x < 5) {
;       goto l;
;     }
;   }
;   // n
;   ret += id << 3;
;   goto p;
;
; l:
;   ret += id << 3;
;
; o:
;   for (int i = 0; i < x / 2; i++) ret += i;
;
; p:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization17(i32 addrspace(1)* %out, i32 noundef %n, i32 noundef %x) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  br label %while.body

while.body:                                       ; preds = %if.end5, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %if.end5 ]
  %cmp = icmp sgt i32 %n, 10
  br i1 %cmp, label %for.cond28, label %if.else

if.else:                                          ; preds = %while.body
  %cmp2 = icmp slt i32 %n, 5
  br i1 %cmp2, label %f, label %if.end5

if.end5:                                          ; preds = %if.else
  %inc = add nuw nsw i32 %i.0, 1
  %rem = and i32 %i.0, 1
  %add = sub nsw i32 0, %rem
  %cmp6 = icmp eq i32 %conv, %add
  br i1 %cmp6, label %for.cond, label %while.body

for.cond:                                         ; preds = %for.body, %if.end5
  %ret.0 = phi i32 [ %inc14, %for.body ], [ 0, %if.end5 ]
  %storemerge = phi i32 [ %inc15, %for.body ], [ 0, %if.end5 ]
  %add11 = add nsw i32 %n, 10
  %cmp12 = icmp slt i32 %storemerge, %add11
  br i1 %cmp12, label %for.body, label %m

for.body:                                         ; preds = %for.cond
  %inc14 = add nuw nsw i32 %ret.0, 1
  %inc15 = add nuw nsw i32 %storemerge, 1
  br label %for.cond

f:                                                ; preds = %if.else
  %div = sdiv i32 %x, 2
  br label %for.cond18

for.cond18:                                       ; preds = %for.body22, %f
  %ret.1 = phi i32 [ %div, %f ], [ %add23, %for.body22 ]
  %storemerge3 = phi i32 [ 0, %f ], [ %inc25, %for.body22 ]
  %div19 = sdiv i32 %x, 2
  %cmp20 = icmp slt i32 %storemerge3, %div19
  br i1 %cmp20, label %for.body22, label %m

for.body22:                                       ; preds = %for.cond18
  %add23 = add nsw i32 %storemerge3, %ret.1
  %inc25 = add nuw nsw i32 %storemerge3, 1
  br label %for.cond18

for.cond28:                                       ; preds = %for.body32, %while.body
  %ret.2 = phi i32 [ %add33, %for.body32 ], [ 0, %while.body ]
  %storemerge4 = phi i32 [ %inc35, %for.body32 ], [ 0, %while.body ]
  %add29 = add nsw i32 %n, 5
  %cmp30 = icmp slt i32 %storemerge4, %add29
  br i1 %cmp30, label %for.body32, label %for.end36

for.body32:                                       ; preds = %for.cond28
  %add33 = add nuw nsw i32 %ret.2, 2
  %inc35 = add nuw nsw i32 %storemerge4, 1
  br label %for.cond28

for.end36:                                        ; preds = %for.cond28
  %rem375 = and i32 %conv, 1
  %cmp38 = icmp eq i32 %rem375, 0
  br i1 %cmp38, label %for.cond43, label %m

m:                                                ; preds = %for.end36, %for.cond18, %for.cond
  %ret.3 = phi i32 [ %ret.0, %for.cond ], [ %ret.1, %for.cond18 ], [ %ret.2, %for.end36 ]
  %shl = shl i32 %ret.3, 2
  br label %o

for.cond43:                                       ; preds = %for.inc52, %for.end36
  %storemerge6 = phi i32 [ %inc53, %for.inc52 ], [ 0, %for.end36 ]
  %div44 = sdiv i32 %x, 2
  %cmp45 = icmp slt i32 %storemerge6, %div44
  br i1 %cmp45, label %for.body47, label %for.end54

for.body47:                                       ; preds = %for.cond43
  %cmp48 = icmp slt i32 %x, 5
  br i1 %cmp48, label %l, label %for.inc52

for.inc52:                                        ; preds = %for.body47
  %inc53 = add nuw nsw i32 %storemerge6, 1
  br label %for.cond43

for.end54:                                        ; preds = %for.cond43
  %shl55 = mul i32 %conv, 8
  %add56 = add nsw i32 %ret.2, %shl55
  br label %p

l:                                                ; preds = %for.body47
  %shl57 = mul i32 %conv, 8
  %add58 = add nsw i32 %ret.2, %shl57
  br label %o

o:                                                ; preds = %l, %m
  %storemerge1 = phi i32 [ %shl, %m ], [ %add58, %l ]
  br label %for.cond60

for.cond60:                                       ; preds = %for.body64, %o
  %ret.4 = phi i32 [ %storemerge1, %o ], [ %add65, %for.body64 ]
  %storemerge2 = phi i32 [ 0, %o ], [ %inc67, %for.body64 ]
  %div61 = sdiv i32 %x, 2
  %cmp62 = icmp slt i32 %storemerge2, %div61
  br i1 %cmp62, label %for.body64, label %p

for.body64:                                       ; preds = %for.cond60
  %add65 = add nsw i32 %storemerge2, %ret.4
  %inc67 = add nuw nsw i32 %storemerge2, 1
  br label %for.cond60

p:                                                ; preds = %for.cond60, %for.end54
  %ret.5 = phi i32 [ %add56, %for.end54 ], [ %ret.4, %for.cond60 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %ret.5, i32 addrspace(1)* %arrayidx, align 4
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
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization17, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization17
; CHECK: br label %[[WHILEBODY:.+]]

; CHECK: [[WHILEBODY]]:
; CHECK: %[[CMP:.+]] = icmp
; CHECK: br i1 %[[CMP]], label %[[FORCOND28PREHEADER:.+]], label %[[IFELSE:.+]]

; CHECK: [[FORCOND28PREHEADER]]:
; CHECK: br label %[[WHILEBODYPUREEXIT:.+]]

; CHECK: [[FORCOND28PREHEADERELSE:.+]]:
; CHECK: br label %[[M:.+]]

; CHECK: [[FORCOND28PREHEADERSPLIT:.+]]:
; CHECK: br label %[[FORCOND28:.+]]

; CHECK: [[IFELSE]]:
; CHECK: %[[CMP2:.+]] = icmp
; CHECK: br i1 %[[CMP2]], label %[[F:.+]], label %[[IFEND5:.+]]

; CHECK: [[IFEND5]]:
; CHECK: br i1 %{{.+}}, label %[[WHILEBODY]], label %[[WHILEBODYPUREEXIT]]

; CHECK: [[WHILEBODYPUREEXIT]]:
; CHECK: br label %[[FORCONDPREHEADER:.+]]

; CHECK: [[FORCONDPREHEADER]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCONDPREHEADERELSE:.+]]:
; CHECK: br i1 %{{.+}}, label %[[FELSE:.+]], label %[[FSPLIT:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[MLOOPEXIT2:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[F]]:
; CHECK: br label %[[WHILEBODYPUREEXIT]]

; CHECK: [[FELSE]]:
; CHECK: br i1 %{{.+}}, label %[[FORCOND28PREHEADERELSE]], label %[[FORCOND28PREHEADERSPLIT]]

; CHECK: [[FSPLIT]]:
; CHECK: br label %[[FORCOND18:.+]]

; CHECK: [[FORCOND18]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY22:.+]], label %[[MLOOPEXIT:.+]]

; CHECK: [[FORBODY22]]:
; CHECK: br label %[[FORCOND18]]

; CHECK: [[FORCOND28]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY32:.+]], label %[[FOREND36:.+]]

; CHECK: [[FORBODY32]]:
; CHECK: br label %[[FORCOND28]]

; CHECK: [[FOREND36]]:
; CHECK: br label %[[FORCOND43PREHEADER:.+]]

; CHECK: [[FORCOND43PREHEADER]]:
; CHECK: br label %[[FORCOND43:.+]]

; CHECK: [[MLOOPEXIT]]:
; CHECK: br label %[[M]]

; CHECK: [[MLOOPEXIT2]]:
; CHECK: br label %[[FORCONDPREHEADERELSE]]

; CHECK: [[M]]:
; CHECK: br label %[[O:.+]]

; CHECK: [[FORCOND43]]:
; CHECK: %[[CMP14:.+]] = icmp
; CHECK: br i1 %[[CMP14]], label %[[FORBODY47:.+]], label %[[FOREND54:.+]]

; CHECK: [[FORBODY47]]:
; CHECK: %[[CMP48:.+]] = icmp
; CHECK: br i1 %[[CMP48]], label %[[L:.+]], label %[[FORINC52:.+]]

; CHECK: [[FORINC52]]:
; CHECK: br label %[[FORCOND43]]

; CHECK: [[FOREND54]]:
; CHECK: br label %[[M]]

; CHECK: [[L]]:
; CHECK: br label %[[M]]

; CHECK: [[O]]:
; CHECK: br label %[[FORCOND60:.+]]

; CHECK: [[FORCOND60]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY64:.+]], label %[[PLOOPEXIT:.+]]

; CHECK: [[FORBODY64]]:
; CHECK: br label %[[FORCOND60]]

; CHECK: [[PLOOPEXIT]]:
; CHECK: br label %[[P:.+]]

; CHECK: [[P]]:
; CHECK: ret void
