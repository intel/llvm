; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: not llvm-spirv %t.bc --spirv-max-version=1.1 -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; Check that UniformId, AlignmentId and MaxByteOffsetId decorations expressed
; as !spirv.Decorations metadata (kinds 27, 46 and 47, per the SPIR-V grammar)
; are emitted as OpDecorateId with an id operand, not as a plain OpDecorate
; with a literal operand.

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

@var = addrspace(1) global i32 0, !spirv.Decorations !1
@var_align = addrspace(1) global i32 0, !spirv.Decorations !3
@var_offset = addrspace(1) global i32 0, !spirv.Decorations !5

; CHECK-DAG: DecorateId [[#TARGET:]] UniformId [[#SCOPE:]]
; CHECK-DAG: DecorateId [[#ALIGN_TARGET:]] AlignmentId [[#ALIGN:]]
; CHECK-DAG: DecorateId [[#OFFSET_TARGET:]] MaxByteOffsetId [[#OFFSET:]]
; CHECK-NOT: Decorate {{[0-9]+}} {{UniformId|AlignmentId|MaxByteOffsetId}}
; CHECK: TypeInt [[#UINT:]] 32 0
; CHECK-DAG: Constant [[#UINT]] [[#SCOPE]] 2
; CHECK-DAG: Constant [[#UINT]] [[#ALIGN]] 8
; CHECK-DAG: Constant [[#UINT]] [[#OFFSET]] 16
; CHECK: Variable {{[0-9]+}} [[#TARGET]]
; CHECK: Variable {{[0-9]+}} [[#ALIGN_TARGET]]
; CHECK: Variable {{[0-9]+}} [[#OFFSET_TARGET]]

; AlignmentId and MaxByteOffsetId require SPIR-V 1.2.
; CHECK-ERROR: RequiresVersion: Cannot fulfill SPIR-V version restriction:
; CHECK-ERROR-NEXT: SPIR-V version was restricted to at most 1.1 (65792) but a construct from the input requires SPIR-V version 1.2 (66048) or above

!1 = !{!2}
!2 = !{i32 27, i32 2}
!3 = !{!4}
!4 = !{i32 46, i32 8}
!5 = !{!6}
!6 = !{i32 47, i32 16}
