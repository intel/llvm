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

; RUN: veczc -k partial_linearization13 -vecz-passes="function(instcombine,simplifycfg),mergereturn,vecz-loop-rotate,function(loop(indvars)),cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; The CFG of the following kernel is:
;
;     a
;    / \
;   b   c
;    \ / \
;     |   \
;     |    d
;     |   / \
;     |  |   e
;     |   \ /
;     |    f
;     |   / \
;     |  |   g
;     |   \ /
;      \   h
;       \ /
;        i
;
; * where nodes d and f are uniform branches, and nodes a and c are varying
;   branches.
; * where nodes b, c, i are divergent.
;
; With BOSCC, it will be transformed as follows:
;
;     a___________
;    / \          \
;   b   c_________ c'
;    \ / \        \|
;     |   \        d'
;     |    d      / \
;     |   / \    |   e'
;     |  |   e    \ /
;     |   \ /      f'
;     |    f      / \
;     |   / \    |   g'
;     |  |   g    \ /
;     |   \ /      h'
;      \   h       |
;       \ /        b'
;        i         |
;        `--> & <- i'
;
; where '&' represents merge blocks of BOSCC regions.
;
; __kernel void partial_linearization13(__global int *out, int n) {
;   size_t tid = get_global_id(0);
;   size_t size = get_global_size(0);
;   // a
;   if (tid + 1 < size) {
;     // b
;     out[tid] = n;
;   } else if (tid + 1 == size) { // c
;     size_t leftovers = 1 + (size & 1);
;     switch (leftovers) { // d
;       case 2: // e
;         out[tid] = 2 * n + 1;
;         // fall through
;       case 1: // f
;         out[tid] += 3 * n - 1;
;         break;
;     }
;     switch (leftovers) { // g
;       case 2:
;         out[tid] /= n;
;         // fall through
;       case 1: // h
;         out[tid]--;
;         break;
;     }
;   }
;   // i
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @partial_linearization13(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %call1 = call i64 @__mux_get_global_size(i32 0) #2
  %add = add i64 %call, 1
  %cmp = icmp ult i64 %add, %call1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %n, i32 addrspace(1)* %arrayidx, align 4
  br label %if.end17

if.else:                                          ; preds = %entry
  %add2 = add i64 %call, 1
  %cmp3 = icmp eq i64 %add2, %call1
  br i1 %cmp3, label %if.then4, label %if.end17

if.then4:                                         ; preds = %if.else
  %0 = and i64 %call1, 1
  %trunc = icmp eq i64 %0, 0
  br i1 %trunc, label %sw.bb8, label %sw.bb

sw.bb:                                            ; preds = %if.then4
  %mul = shl nsw i32 %n, 1
  %add6 = or i32 %mul, 1
  %arrayidx7 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %add6, i32 addrspace(1)* %arrayidx7, align 4
  br label %sw.bb8

sw.bb8:                                           ; preds = %sw.bb, %if.then4
  %mul9 = mul nsw i32 %n, 3
  %sub = add nsw i32 %mul9, -1
  %arrayidx10 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  %1 = load i32, i32 addrspace(1)* %arrayidx10, align 4
  %add11 = add nsw i32 %sub, %1
  store i32 %add11, i32 addrspace(1)* %arrayidx10, align 4
  %2 = and i64 %call1, 1
  %trunc2 = icmp ne i64 %2, 0
  %trunc2.off = add i1 %trunc2, true
  %switch = icmp ult i1 %trunc2.off, true
  br i1 %switch, label %sw.bb12, label %sw.bb14

sw.bb12:                                          ; preds = %sw.bb8
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  %3 = load i32, i32 addrspace(1)* %arrayidx13, align 4
  %4 = icmp eq i32 %3, -2147483648
  %5 = icmp eq i32 %n, -1
  %6 = and i1 %5, %4
  %7 = icmp eq i32 %n, 0
  %8 = or i1 %7, %6
  %9 = select i1 %8, i32 1, i32 %n
  %div = sdiv i32 %3, %9
  store i32 %div, i32 addrspace(1)* %arrayidx13, align 4
  br label %sw.bb14

sw.bb14:                                          ; preds = %sw.bb12, %sw.bb8
  %arrayidx15 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  %10 = load i32, i32 addrspace(1)* %arrayidx15, align 4
  %dec = add nsw i32 %10, -1
  store i32 %dec, i32 addrspace(1)* %arrayidx15, align 4
  br label %if.end17

if.end17:                                         ; preds = %sw.bb14, %if.else, %if.then
  ret void
}

; Function Attrs: nounwind readonly
declare i64 @__mux_get_global_id(i32) #1

; Function Attrs: nounwind readonly
declare i64 @__mux_get_global_size(i32) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.kernels = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{void (i32 addrspace(1)*, i32)* @partial_linearization13, !3, !4, !5, !6, !7, !8}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !""}
!8 = !{!"kernel_arg_name", !"out", !"n"}

; CHECK: spir_kernel void @__vecz_v4_partial_linearization13
; CHECK: br i1 %{{.+}}, label %[[IFTHENUNIFORM:.+]], label %[[ENTRYBOSCCINDIR:.+]]

; CHECK: [[IFELSEUNIFORM:.+]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN4UNIFORM:.+]], label %[[IFELSEUNIFORMBOSCCINDIR:.+]]

; CHECK: [[IFTHEN4UNIFORM]]:
; CHECK: %[[TRUNCUNIFORM:.+]] = icmp
; CHECK: br i1 %[[TRUNCUNIFORM]], label %[[SWBB8UNIFORM:.+]], label %[[SWBBUNIFORM:.+]]

; CHECK: [[IFELSEUNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFEND17UNIFORM:.+]], label %[[IFTHEN4:.+]]

; CHECK: [[SWBBUNIFORM]]:
; CHECK: br label %[[SWBB8UNIFORM]]

; CHECK: [[SWBB8UNIFORM]]:
; CHECK: %[[TRUNC2UNIFORM:.+]] = icmp
; CHECK: br i1 %[[TRUNC2UNIFORM]], label %[[SWBB14UNIFORM:.+]], label %[[SWBB12UNIFORM:.+]]

; CHECK: [[SWBB12UNIFORM]]:
; CHECK: br label %[[SWBB14UNIFORM]]

; CHECK: [[SWBB14UNIFORM]]:
; CHECK: br label %[[IFEND17UNIFORM]]

; CHECK: [[IFTHENUNIFORM]]:
; CHECK: br label %[[IFEND17:.+]]

; CHECK: [[ENTRYBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFELSEUNIFORM]], label %[[IFELSE:.+]]

; CHECK: [[IFTHEN:.+]]:
; CHECK: br label %[[IFEND17]]

; CHECK: [[IFELSE]]:
; CHECK: br label %[[IFTHEN4]]

; CHECK: [[IFTHEN4]]:
; CHECK: %[[TRUNC:.+]] = icmp
; FIXME: We shouldn't need to mask this comparison, as it's truly uniform even
; on inactive lanes.
; CHECK: %[[TRUNC_ACTIVE:.+]] = and i1 %[[TRUNC]], {{%.*}}
; CHECK: %[[TRUNC_ACTIVE_ANY:.+]] = call i1 @__vecz_b_divergence_any(i1 %[[TRUNC_ACTIVE]])
; CHECK: br i1 %[[TRUNC_ACTIVE_ANY]], label %[[SWBB8:.+]], label %[[SWBB:.+]]

; CHECK: [[SWBB]]:
; CHECK: br label %[[SWBB8]]

; CHECK: [[SWBB8]]:
; CHECK: %[[TRUNC2:.+]] = icmp
; CHECK: br i1 %[[TRUNC2]], label %[[SWBB14:.+]], label %[[SWBB12:.+]]

; CHECK: [[SWBB12]]:
; CHECK: br label %[[SWBB14]]

; CHECK: [[SWBB14]]:
; CHECK: br label %[[IFTHEN]]

; CHECK: [[IFEND17]]:
; CHECK: ret void
