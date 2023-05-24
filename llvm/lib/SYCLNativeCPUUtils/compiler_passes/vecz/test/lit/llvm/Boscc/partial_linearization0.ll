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

; RUN: %veczc -k partial_linearization0 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | %filecheck %s

; The CFG of the following kernel is:
;
;        a
;       / \
;      b   c
;       \ /
;        d
;        |
;        e
;       / \
;      /   \
;     f     g
;    / \   / \
;   h   i j   k
;    \ /   \ /
;     l     m
;      \   /
;       \ /
;        n
;
; * where node e is a uniform branch, and nodes a, f and g are varying
;   branches.
; * where nodes b, c, d, h, i, j, k, l, m are divergent.
;
; With BOSCC, it will be transformed as follows:
;
;         a___
;        / \  \
;       b   c  c'
;        \ /   |
;         d    b'
;         |    |
;         |    d'
;         |   /
;          \ /
;           e
;          / \
;         /   \
;     ___f     g___
;    /  / \   / \  \
;   i' h   i j   k  k'
;   |   \ /   \ /   |
;   h'   l     m    j'
;   |    |     |    |
;   l'   |     |    m'
;    \   |     |   /
;     \ /       \ /
;      & -> n <- &
;
; where '&' represents merge blocks of BOSCC regions.
;
; __kernel void partial_linearization0(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   if (id % 5 == 0) {
;     for (int i = 0; i < n * 2; i++) ret++;
;   } else {
;     for (int i = 0; i < n / 4; i++) ret++;
;   }
;
;   if (n > 10) { // uniform
;     if (id % 2 == 0) { // varying
;       for (int i = 0; i < n + 10; i++) ret++;
;     } else { // varying
;       for (int i = 0; i < n + 10; i++) ret *= 2;
;     }
;     ret += id * 10;
;   } else { // uniform
;     if (id % 2 == 0) { // varying
;       for (int i = 0; i < n + 8; i++) ret++;
;     } else { // varying
;       for (int i = 0; i < n + 8; i++) ret *= 2;
;     }
;     ret += id / 2;
;   }
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization0(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %conv = trunc i64 %call to i32
  %rem = srem i32 %conv, 5
  %cmp = icmp eq i32 %rem, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.then
  %ret.0 = phi i32 [ 0, %if.then ], [ %inc, %for.body ]
  %storemerge8 = phi i32 [ 0, %if.then ], [ %inc4, %for.body ]
  %mul = shl nsw i32 %n, 1
  %cmp2 = icmp slt i32 %storemerge8, %mul
  br i1 %cmp2, label %for.body, label %if.end

for.body:                                         ; preds = %for.cond
  %inc = add nsw i32 %ret.0, 1
  %inc4 = add nsw i32 %storemerge8, 1
  br label %for.cond

if.else:                                          ; preds = %entry
  br label %for.cond6

for.cond6:                                        ; preds = %for.body9, %if.else
  %ret.1 = phi i32 [ 0, %if.else ], [ %inc10, %for.body9 ]
  %storemerge = phi i32 [ 0, %if.else ], [ %inc12, %for.body9 ]
  %div = sdiv i32 %n, 4
  %cmp7 = icmp slt i32 %storemerge, %div
  br i1 %cmp7, label %for.body9, label %if.end

for.body9:                                        ; preds = %for.cond6
  %inc10 = add nsw i32 %ret.1, 1
  %inc12 = add nsw i32 %storemerge, 1
  br label %for.cond6

if.end:                                           ; preds = %for.cond6, %for.cond
  %ret.2 = phi i32 [ %ret.0, %for.cond ], [ %ret.1, %for.cond6 ]
  %cmp14 = icmp sgt i32 %n, 10
  %rem175 = and i32 %conv, 1
  %cmp18 = icmp eq i32 %rem175, 0
  br i1 %cmp14, label %if.then16, label %if.else44

if.then16:                                        ; preds = %if.end
  br i1 %cmp18, label %if.then20, label %if.else30

if.then20:                                        ; preds = %if.then16
  br label %for.cond22

for.cond22:                                       ; preds = %for.body25, %if.then20
  %ret.3 = phi i32 [ %ret.2, %if.then20 ], [ %inc26, %for.body25 ]
  %storemerge7 = phi i32 [ 0, %if.then20 ], [ %inc28, %for.body25 ]
  %add = add nsw i32 %n, 10
  %cmp23 = icmp slt i32 %storemerge7, %add
  br i1 %cmp23, label %for.body25, label %if.end41

for.body25:                                       ; preds = %for.cond22
  %inc26 = add nsw i32 %ret.3, 1
  %inc28 = add nsw i32 %storemerge7, 1
  br label %for.cond22

if.else30:                                        ; preds = %if.then16
  br label %for.cond32

for.cond32:                                       ; preds = %for.body36, %if.else30
  %ret.4 = phi i32 [ %ret.2, %if.else30 ], [ %mul37, %for.body36 ]
  %storemerge6 = phi i32 [ 0, %if.else30 ], [ %inc39, %for.body36 ]
  %add33 = add nsw i32 %n, 10
  %cmp34 = icmp slt i32 %storemerge6, %add33
  br i1 %cmp34, label %for.body36, label %if.end41

for.body36:                                       ; preds = %for.cond32
  %mul37 = shl nsw i32 %ret.4, 1
  %inc39 = add nsw i32 %storemerge6, 1
  br label %for.cond32

if.end41:                                         ; preds = %for.cond32, %for.cond22
  %ret.5 = phi i32 [ %ret.3, %for.cond22 ], [ %ret.4, %for.cond32 ]
  %mul42 = mul nsw i32 %conv, 10
  %add43 = add nsw i32 %ret.5, %mul42
  br label %if.end73

if.else44:                                        ; preds = %if.end
  br i1 %cmp18, label %if.then48, label %if.else59

if.then48:                                        ; preds = %if.else44
  br label %for.cond50

for.cond50:                                       ; preds = %for.body54, %if.then48
  %ret.6 = phi i32 [ %ret.2, %if.then48 ], [ %inc55, %for.body54 ]
  %storemerge4 = phi i32 [ 0, %if.then48 ], [ %inc57, %for.body54 ]
  %add51 = add nsw i32 %n, 8
  %cmp52 = icmp slt i32 %storemerge4, %add51
  br i1 %cmp52, label %for.body54, label %if.end70

for.body54:                                       ; preds = %for.cond50
  %inc55 = add nsw i32 %ret.6, 1
  %inc57 = add nsw i32 %storemerge4, 1
  br label %for.cond50

if.else59:                                        ; preds = %if.else44
  br label %for.cond61

for.cond61:                                       ; preds = %for.body65, %if.else59
  %ret.7 = phi i32 [ %ret.2, %if.else59 ], [ %mul66, %for.body65 ]
  %storemerge2 = phi i32 [ 0, %if.else59 ], [ %inc68, %for.body65 ]
  %add62 = add nsw i32 %n, 8
  %cmp63 = icmp slt i32 %storemerge2, %add62
  br i1 %cmp63, label %for.body65, label %if.end70

for.body65:                                       ; preds = %for.cond61
  %mul66 = shl nsw i32 %ret.7, 1
  %inc68 = add nsw i32 %storemerge2, 1
  br label %for.cond61

if.end70:                                         ; preds = %for.cond61, %for.cond50
  %ret.8 = phi i32 [ %ret.6, %for.cond50 ], [ %ret.7, %for.cond61 ]
  %div71 = sdiv i32 %conv, 2
  %add72 = add nsw i32 %ret.8, %div71
  br label %if.end73

if.end73:                                         ; preds = %if.end70, %if.end41
  %storemerge3 = phi i32 [ %add72, %if.end70 ], [ %add43, %if.end41 ]
  %idxprom = sext i32 %conv to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %storemerge3, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.kernels = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization0, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization0
; CHECK: br i1 %{{.+}}, label %[[FORCONDPREHEADERUNIFORM:.+]], label %[[ENTRYBOSCCINDIR:.+]]

; CHECK: [[FORCOND6PREHEADERUNIFORM:.+]]:
; CHECK: br label %[[FORCOND6UNIFORM:.+]]

; CHECK: [[FORCOND6UNIFORM]]:
; CHECK: %[[CMP7UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP7UNIFORM]], label %[[FORBODY9UNIFORM:.+]], label %[[IFENDLOOPEXIT3UNIFORM:.+]]

; CHECK: [[FORBODY9UNIFORM]]:
; CHECK: br label %[[FORCOND6UNIFORM]]

; CHECK: [[IFENDLOOPEXIT3UNIFORM]]:
; CHECK: br label %[[IFEND:.+]]

; CHECK: [[FORCONDPREHEADERUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM:.+]]

; CHECK: [[ENTRYBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[FORCOND6PREHEADERUNIFORM]], label %[[FORCOND6PREHEADER:.+]]

; CHECK: [[FORCONDUNIFORM]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODYUNIFORM:.+]], label %[[IFENDLOOPEXITUNIFORM:.+]]

; CHECK: [[FORBODYUNIFORM]]:
; CHECK: br label %[[FORCONDUNIFORM]]

; CHECK: [[IFENDLOOPEXITUNIFORM]]:
; CHECK: br label %[[IFEND]]

; CHECK: [[FORCOND6PREHEADER]]:
; CHECK: br label %[[FORCOND6:.+]]

; CHECK: [[FORCONDPREHEADER:.+]]:
; CHECK: br label %[[FORCOND:.+]]

; CHECK: [[FORCOND]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY:.+]], label %[[IFENDLOOPEXIT:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND]]

; CHECK: [[FORCOND6]]:
; CHECK: %[[CMP7:.+]] = icmp
; CHECK: br i1 %[[CMP7]], label %[[FORBODY9:.+]], label %[[IFENDLOOPEXIT3:.+]]

; CHECK: [[FORBODY9]]:
; CHECK: br label %[[FORCOND6]]

; CHECK: [[IFENDLOOPEXIT]]:
; CHECK: br label %[[IFEND]]

; CHECK: [[IFENDLOOPEXIT3]]:
; CHECK: br label %[[FORCONDPREHEADER]]

; CHECK: [[IFEND]]:
; CHECK: %[[CMP14:.+]] = icmp
; CHECK: br i1 %[[CMP14]], label %[[IFTHEN16:.+]], label %[[IFELSE44:.+]]

; CHECK: [[IFTHEN16]]:
; CHECK: br i1 %{{.+}}, label %[[FORCOND22PREHEADERUNIFORM:.+]], label %[[IFTHEN16BOSCCINDIR:.+]]

; CHECK: [[FORCOND32PREHEADERUNIFORM:.+]]:
; CHECK: br label %[[FORCOND32UNIFORM:.+]]

; CHECK: [[FORCOND32UNIFORM]]:
; CHECK: %[[CMP34UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP34UNIFORM]], label %[[FORBODY36UNIFORM:.+]], label %[[IFEND41LOOPEXIT1UNIFORM:.+]]

; CHECK: [[FORBODY36UNIFORM]]:
; CHECK: br label %[[FORCOND32UNIFORM]]

; CHECK: [[IFEND41LOOPEXIT1UNIFORM]]:
; CHECK: br label %[[IFEND41UNIFORM:.+]]

; CHECK: [[FORCOND22PREHEADERUNIFORM]]:
; CHECK: br label %[[FORCOND22UNIFORM:.+]]

; CHECK: [[IFTHEN16BOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[FORCOND32PREHEADERUNIFORM]], label %[[FORCOND32PREHEADER:.+]]

; CHECK: [[FORCOND22UNIFORM]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY25UNIFORM:.+]], label %[[IFEND41LOOPEXITUNIFORM:.+]]

; CHECK: [[FORBODY25UNIFORM]]:
; CHECK: br label %[[FORCOND22UNIFORM]]

; CHECK: [[IFEND41LOOPEXITUNIFORM]]:
; CHECK: br label %[[IFEND41:.+]]

; CHECK: [[FORCOND32PREHEADER]]:
; CHECK: br label %[[FORCOND32:.+]]

; CHECK: [[FORCOND22PREHEADER:.+]]:
; CHECK: br label %[[FORCOND22:.+]]

; CHECK: [[FORCOND22]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY25:.+]], label %[[IFEND41LOOPEXIT:.+]]

; CHECK: [[FORBODY25]]:
; CHECK: br label %[[FORCOND22]]

; CHECK: [[FORCOND32]]:
; CHECK: %[[CMP34:.+]] = icmp
; CHECK: br i1 %[[CMP34]], label %[[FORBODY36:.+]], label %[[IFEND41LOOPEXIT1:.+]]

; CHECK: [[FORBODY36]]:
; CHECK: br label %[[FORCOND32]]

; CHECK: [[IFEND41LOOPEXIT]]:
; CHECK: br label %[[IFEND41]]

; CHECK: [[IFEND41LOOPEXIT1]]:
; CHECK: br label %[[FORCOND22PREHEADER]]

; CHECK: [[IFEND41]]:
; CHECK: br label %[[IFEND73:.+]]

; CHECK: [[IFELSE44]]:
; CHECK: br i1 %{{.+}}, label %[[FORCOND50PREHEADERUNIFORM:.+]], label %[[IFELSE44BOSCCINDIR:.+]]

; CHECK: [[FORCOND61PREHEADERUNIFORM:.+]]:
; CHECK: br label %[[FORCOND61UNIFORM:.+]]

; CHECK: [[FORCOND61UNIFORM]]:
; CHECK: %[[CMP63UNIFORM:.+]] = icmp
; CHECK: br i1 %[[CMP63UNIFORM]], label %[[FORBODY65UNIFORM:.+]], label %[[IFEND70LOOPEXIT2UNIFORM:.+]]

; CHECK: [[FORBODY65UNIFORM]]:
; CHECK: br label %[[FORCOND61UNIFORM]]

; CHECK: [[IFEND70LOOPEXIT2UNIFORM]]:
; CHECK: br label %[[IFEND70UNIFORM:.+]]

; CHECK: [[FORCOND50PREHEADERUNIFORM]]:
; CHECK: br label %[[FORCOND50UNIFORM:.+]]

; CHECK: [[IFELSE44BOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[FORCOND61PREHEADERUNIFORM]], label %[[FORCOND61PREHEADER:.+]]

; CHECK: [[FORCOND50UNIFORM]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY54UNIFORM:.+]], label %[[IFEND70LOOPEXITUNIFORM:.+]]

; CHECK: [[FORBODY54UNIFORM]]:
; CHECK: br label %[[FORCOND50UNIFORM]]

; CHECK: [[IFEND70LOOPEXITUNIFORM]]:
; CHECK: br label %[[IFEND70:.+]]

; CHECK: [[FORCOND61PREHEADER]]:
; CHECK: br label %[[FORCOND61:.+]]

; CHECK: [[FORCOND50PREHEADER:.+]]:
; CHECK: br label %[[FORCOND50:.+]]

; CHECK: [[FORCOND50]]:
; CHECK: br i1 {{(%[0-9A-Za-z\.]+)|(false)}}, label %[[FORBODY54:.+]], label %[[IFEND70LOOPEXIT:.+]]

; CHECK: [[FORBODY54]]:
; CHECK: br label %[[FORCOND50]]

; CHECK: [[FORCOND61]]:
; CHECK: %[[CMP63:.+]] = icmp
; CHECK: br i1 %[[CMP63]], label %[[FORBODY65:.+]], label %[[IFEND70LOOPEXIT2:.+]]

; CHECK: [[FORBODY65]]:
; CHECK: br label %[[FORCOND61]]

; CHECK: [[IFEND70LOOPEXIT]]:
; CHECK: br label %[[IFEND70]]

; CHECK: [[IFEND70LOOPEXIT2]]:
; CHECK: br label %[[FORCOND50PREHEADER]]

; CHECK: [[IFEND70]]:
; CHECK: br label %[[IFEND73]]

; CHECK: [[IFEND73]]:
; CHECK: ret void
