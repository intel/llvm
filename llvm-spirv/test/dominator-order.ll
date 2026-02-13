; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t
; RUN: FileCheck < %t %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; This test checks that basic blocks are reordered in SPIR-V so that dominators
; are emitted ahead of their dominated blocks as required by the SPIR-V
; specification.

; CHECK-DAG: Name [[#ENTRY:]] "entry"
; CHECK-DAG: Name [[#FOR_BODY137_LR_PH:]] "for.body137.lr.ph"
; CHECK-DAG: Name [[#FOR_BODY:]] "for.body"

; CHECK: Label [[#ENTRY]]
; CHECK: Label [[#FOR_BODY]]
; CHECK: Label [[#FOR_BODY137_LR_PH]]

source_filename = "reproducer.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr addrspace(1) %arg) local_unnamed_addr #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
entry:
  br label %for.body

for.body137.lr.ph:                                ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.body, label %for.body137.lr.ph
}

attributes #0 = { "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.ident = !{!0}

!0 = !{!"clang version 9.0.0 "}
!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"uchar*"}
!4 = !{!""}
