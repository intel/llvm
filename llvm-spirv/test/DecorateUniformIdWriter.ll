; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; Check that a UniformId decoration expressed as !spirv.Decorations metadata
; (kind 27, per the SPIR-V grammar) is emitted as OpDecorateId with a Scope
; id operand, not as a plain OpDecorate with a literal operand.

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

@var = addrspace(1) global i32 0, !spirv.Decorations !1

; CHECK: DecorateId [[#TARGET:]] UniformId [[#SCOPE:]]
; CHECK-NOT: Decorate {{[0-9]+}} UniformId
; CHECK: TypeInt [[#UINT:]] 32 0
; CHECK: Constant [[#UINT]] [[#SCOPE]] 2
; CHECK: Variable {{[0-9]+}} [[#TARGET]]

!1 = !{!2}
!2 = !{i32 27, i32 2}
