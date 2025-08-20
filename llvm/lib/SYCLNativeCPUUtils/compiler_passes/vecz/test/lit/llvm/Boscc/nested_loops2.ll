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

; RUN: veczc -k nested_loops2 -vecz-passes=vecz-loop-rotate,cfg-convert -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
source_filename = "kernel.opencl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @nested_loops2(i32 addrspace(1)* %out, i32 %n) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %conv = trunc i64 %call to i32
  %cmp = icmp slt i32 %conv, 16
  br i1 %cmp, label %if.then, label %if.end25

if.then:                                          ; preds = %entry
  %mul2 = mul nsw i32 %conv, %n
  %0 = icmp eq i32 %mul2, -2147483648
  %1 = icmp eq i32 %n, -1
  %2 = and i1 %1, %0
  %3 = icmp eq i32 %n, 0
  %4 = or i1 %3, %2
  %5 = select i1 %4, i32 1, i32 %n
  %div3 = sdiv i32 %mul2, %5
  %add = add nsw i32 %div3, %conv
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %ret.0 = phi i32 [ 0, %if.then ], [ %ret.2, %for.inc ]
  %storemerge = phi i32 [ 0, %if.then ], [ %inc24, %for.inc ]
  %cmp7 = icmp slt i32 %storemerge, %n
  br i1 %cmp7, label %for.body, label %if.end25

for.body:                                         ; preds = %for.cond
  %cmp9 = icmp slt i32 %conv, 9
  br i1 %cmp9, label %while.body, label %for.inc

while.body:                                       ; preds = %while.body, %for.body
  %ret.1 = phi i32 [ %ret.0, %for.body ], [ %add17, %while.body ]
  %j.0 = phi i32 [ 0, %for.body ], [ %inc18, %while.body ]
  %mul13 = mul nsw i32 %mul2, %mul2
  %6 = icmp eq i32 %n, 0
  %7 = select i1 %6, i32 1, i32 %n
  %div14 = sdiv i32 %mul13, %7
  %reass.add = add i32 %div14, %add
  %reass.mul = mul i32 %reass.add, 8
  %add6 = add i32 %mul2, 1
  %add16 = add i32 %add6, %add
  %inc = add i32 %add16, %ret.1
  %add17 = add i32 %inc, %reass.mul
  %inc18 = add nuw nsw i32 %j.0, 1
  %add19 = add nsw i32 %j.0, %conv
  %cmp20 = icmp sgt i32 %add19, 3
  br i1 %cmp20, label %for.inc, label %while.body

for.inc:                                          ; preds = %for.body, %while.body
  %ret.2 = phi i32 [ %ret.0, %for.body ], [ %add17, %while.body ]
  %inc24 = add nuw nsw i32 %storemerge, 1
  br label %for.cond

if.end25:                                         ; preds = %for.cond, %entry
  %ret.3 = phi i32 [ 0, %entry ], [ %ret.0, %for.cond ]
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
!llvm.ident = !{!2}
!opencl.kernels = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 8.0.0 (https://github.com/llvm-mirror/clang.git bfbe338a893dde6ba65b2bed6ffea1652a592819) (https://github.com/llvm-mirror/llvm.git a99d6d2122ca2f208e1c4bcaf02ff5930f244f34)"}
!3 = !{void (i32 addrspace(1)*, i32)* @nested_loops2, !4, !5, !6, !7, !8, !9}
!4 = !{!"kernel_arg_addr_space", i32 1, i32 0}
!5 = !{!"kernel_arg_access_qual", !"none", !"none"}
!6 = !{!"kernel_arg_type", !"int*", !"int"}
!7 = !{!"kernel_arg_base_type", !"int*", !"int"}
!8 = !{!"kernel_arg_type_qual", !"", !""}
!9 = !{!"kernel_arg_name", !"out", !"n"}

; The purpose of this test is to make sure we correctly add a boscc connection
; at a div causing latch from the uniform region.

; CHECK: spir_kernel void @__vecz_v4_nested_loops2
; CHECK: entry:
; CHECK: %[[BOSCC:.+]] = call i1 @__vecz_b_divergence_all(i1 %cmp)
; CHECK: br i1 %[[BOSCC]], label %if.then.uniform, label %entry.boscc_indir

; CHECK: if.then.uniform:
; CHECK: br i1 %cmp71.uniform, label %for.body.lr.ph.uniform, label %if.end25.loopexit.uniform

; CHECK: entry.boscc_indir:
; CHECK: %[[BOSCC2:.+]] = call i1 @__vecz_b_divergence_all(i1 %cmp.not{{.*}})
; CHECK: br i1 %[[BOSCC2]], label %if.end25, label %if.then

; CHECK: for.body.lr.ph.uniform:
; CHECK: br label %for.body.uniform

; CHECK: for.body.uniform:
; CHECK: br i1 %[[LBLCOND:.+]], label %while.body.preheader.uniform, label %for.body.uniform.boscc_indir

; CHECK: while.body.preheader.uniform:
; CHECK: br label %while.body.uniform

; CHECK: for.body.uniform.boscc_indir:
; CHECK: %[[BOSCC3:.+]] = call i1 @__vecz_b_divergence_all(i1 %for.inc.uniform.exit_mask)
; CHECK: br i1 %[[BOSCC3]], label %for.inc.uniform, label %for.body.uniform.boscc_store

; CHECK: while.body.uniform:
; CHECK: %cmp20.uniform = icmp sgt i32 %add19.uniform, 3
; CHECK-NOT: br i1 %[[LBLCOND3:.+]], label %for.inc.loopexit.uniform, label %while.body.uniform
; CHECK: br i1 %[[LBLCOND2:.+]], label %for.inc.loopexit.uniform, label %while.body.uniform.boscc_indir
