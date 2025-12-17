; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

declare <2 x half> @llvm.rint.v2f16(<2 x half>)
declare <2 x half> @llvm.nearbyint.v2f16(<2 x half>)
declare <2 x half> @llvm.floor.v2f16(<2 x half>)
declare <2 x half> @llvm.round.v2f16(<2 x half>)
declare <2 x half> @llvm.trunc.v2f16(<2 x half>)
declare <2 x half> @llvm.sin.v2f16(<2 x half>)
declare <2 x half> @llvm.cos.v2f16(<2 x half>)
declare <2 x half> @llvm.exp2.v2f16(<2 x half>)
declare <2 x half> @llvm.log.v2f16(<2 x half>)
declare <2 x half> @llvm.log10.v2f16(<2 x half>)
declare <2 x half> @llvm.log2.v2f16(<2 x half>)

; CHECK-DAG: Name [[#VECTOR_RINT:]] "vector_rint"
; CHECK-DAG: Name [[#VECTOR_NEARBYINT:]] "vector_nearbyint"
; CHECK-DAG: Name [[#VECTOR_FLOOR:]] "vector_floor"
; CHECK-DAG: Name [[#VECTOR_ROUND:]] "vector_round"
; CHECK-DAG: Name [[#VECTOR_TRUNC:]] "vector_trunc"
; CHECK-DAG: Name [[#VECTOR_SIN:]] "vector_sin"
; CHECK-DAG: Name [[#VECTOR_COS:]] "vector_cos"
; CHECK-DAG: Name [[#VECTOR_EXP2:]] "vector_exp2"
; CHECK-DAG: Name [[#VECTOR_LOG:]] "vector_log"
; CHECK-DAG: Name [[#VECTOR_LOG10:]] "vector_log10"
; CHECK-DAG: Name [[#VECTOR_LOG2:]] "vector_log2"

; CHECK-DAG: ExtInstImport [[#CLEXT:]] "OpenCL.std"

; CHECK: Function [[#]] [[#VECTOR_RINT]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] rint [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_rint(<2 x half> %a) {
    %r = call <2 x half> @llvm.rint.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_NEARBYINT]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] rint [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_nearbyint(<2 x half> %a) {
    %r = call <2 x half> @llvm.nearbyint.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_FLOOR]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] floor [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_floor(<2 x half> %a) {
    %r = call <2 x half> @llvm.floor.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_ROUND]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] round [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_round(<2 x half> %a) {
    %r = call <2 x half> @llvm.round.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_TRUNC]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] trunc [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_trunc(<2 x half> %a) {
    %r = call <2 x half> @llvm.trunc.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_SIN]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] sin [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_sin(<2 x half> %a) {
    %r = call <2 x half> @llvm.sin.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_COS]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] cos [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_cos(<2 x half> %a) {
    %r = call <2 x half> @llvm.cos.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_EXP2]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] exp2 [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_exp2(<2 x half> %a) {
    %r = call <2 x half> @llvm.exp2.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_LOG]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] log [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_log(<2 x half> %a) {
    %r = call <2 x half> @llvm.log.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_LOG10]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] log10 [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_log10(<2 x half> %a) {
    %r = call <2 x half> @llvm.log10.v2f16(<2 x half> %a)
    ret <2 x half> %r
}

; CHECK: Function [[#]] [[#VECTOR_LOG2]] [[#]] [[#]] 
; CHECK-NEXT: FunctionParameter [[#]] [[#A:]]
; CHECK: Label
; CHECK: ExtInst [[#]] [[#R:]] [[#CLEXT]] log2 [[#A]]
; CHECK: ReturnValue [[#R]]
; CHECK: FunctionEnd

define <2 x half> @vector_log2(<2 x half> %a) {
    %r = call <2 x half> @llvm.log2.v2f16(<2 x half> %a)
    ret <2 x half> %r
}
