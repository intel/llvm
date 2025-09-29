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

; RUN: veczc -k codegen_2 -vecz-simd-width 16 -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @codegen_2(i32 addrspace(1)* nocapture readonly %in, i32 addrspace(1)* nocapture %out, i32 %size, i32 %reps) local_unnamed_addr {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %conv = sext i32 %reps to i64
  %mul = mul i64 %call, %conv
  %add = add i64 %call, 1
  %mul2 = mul i64 %add, %conv
  %cmp19 = icmp ult i64 %mul, %mul2
  br i1 %cmp19, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %conv4 = sext i32 %size to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %sum.1, %for.inc ]
  %arrayidx8 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %sum.0.lcssa, i32 addrspace(1)* %arrayidx8, align 4, !tbaa !9
  ret void

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %i.021 = phi i64 [ %mul, %for.body.lr.ph ], [ %inc, %for.inc ]
  %sum.020 = phi i32 [ 0, %for.body.lr.ph ], [ %sum.1, %for.inc ]
  %cmp5 = icmp ult i64 %i.021, %conv4
  br i1 %cmp5, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %i.021
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4, !tbaa !9
  %add7 = add nsw i32 %0, %sum.020
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %sum.1 = phi i32 [ %add7, %if.then ], [ %sum.020, %for.body ]
  %inc = add nuw i64 %i.021, 1
  %cmp = icmp ult i64 %inc, %mul2
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

declare i64 @__mux_get_global_id(i32) local_unnamed_addr

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.kernels = !{!2}
!host.build_options = !{!8}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{void (i32 addrspace(1)*, i32 addrspace(1)*, i32, i32)* @codegen_2, !3, !4, !5, !6, !7}
!3 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 0, i32 0}
!4 = !{!"kernel_arg_access_qual", !"none", !"none", !"none", !"none"}
!5 = !{!"kernel_arg_type", !"int*", !"int*", !"int", !"int"}
!6 = !{!"kernel_arg_base_type", !"int*", !"int*", !"int", !"int"}
!7 = !{!"kernel_arg_type_qual", !"const", !"", !"", !""}
!8 = !{!""}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}


; It checks that the PHI node did not prevent the interleave factor from being determined
; CHECK: define spir_kernel void @__vecz_v16_codegen_2
; CHECK-NOT: call <16 x i32> @__vecz_b_masked_gather_load4_4_Dv16_jDv16_u3ptrU3AS1Dv16_b
; CHECK: call <16 x i32> @__vecz_b_masked_interleaved_load4_V_Dv16_ju3ptrU3AS1Dv16_b
