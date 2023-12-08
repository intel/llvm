; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK-SPIRV: TypeBool [[#BoolTy:]]
; CHECK-SPIRV: TypeVector [[#VecTy:]] [[#BoolTy]] 4
; CHECK-SPIRV: ConstantNull [[#VecTy]] [[#VecNullVal:]]
; CHECK-SPIRV: ReturnValue [[#VecNullVal]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func <4 x i1> @f(<4 x float> %0) {
 %2 = fcmp false <4 x float> %0, %0
 ret <4 x i1> %2
}
