; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-SPIRV-DAG: Name [[#VECTOR_SHL:]] "vector_shl"
; CHECK-SPIRV-DAG: Name [[#VECTOR_LSHR:]] "vector_lshr"
; CHECK-SPIRV-DAG: Name [[#VECTOR_ASHR:]] "vector_ashr"
; CHECK-SPIRV-DAG: Name [[#VECTOR_AND:]] "vector_and"
; CHECK-SPIRV-DAG: Name [[#VECTOR_OR:]] "vector_or"
; CHECK-SPIRV-DAG: Name [[#VECTOR_XOR:]] "vector_xor"

; CHECK-SPIRV-NOT: DAG-FENCE

; CHECK-SPIRV-DAG: TypeInt [[#I16:]] 16
; CHECK-SPIRV-DAG: TypeVector [[#VECTOR:]] [[#I16]]
; CHECK-SPIRV-DAG: TypeFunction [[#VECTOR_FN:]] [[#VECTOR]] [[#VECTOR]] [[#VECTOR]]

;; Test shl on vector:
define <2 x i16> @vector_shl(<2 x i16> %a, <2 x i16> %b) {
    %c = shl <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV:      ShiftLeftLogical [[#VECTOR]] [[#C:]] [[#A]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = shl <2 x i16> %{{.*}}, %{{.*}}

;; Test lshr on vector:
define <2 x i16> @vector_lshr(<2 x i16> %a, <2 x i16> %b) {
    %c = lshr <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV:      ShiftRightLogical [[#VECTOR]] [[#C:]] [[#A]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = lshr <2 x i16> %{{.*}}, %{{.*}}

;; Test ashr on vector:
define <2 x i16> @vector_ashr(<2 x i16> %a, <2 x i16> %b) {
    %c = ashr <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV:      ShiftRightArithmetic [[#VECTOR]] [[#C:]] [[#A]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = ashr <2 x i16> %{{.*}}, %{{.*}}

;; Test and on vector:
define <2 x i16> @vector_and(<2 x i16> %a, <2 x i16> %b) {
    %c = and <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV:      BitwiseAnd [[#VECTOR]] [[#C:]] [[#A]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = and <2 x i16> %{{.*}}, %{{.*}}

;; Test or on vector:
define <2 x i16> @vector_or(<2 x i16> %a, <2 x i16> %b) {
    %c = or <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV:      BitwiseOr [[#VECTOR]] [[#C:]] [[#A]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = or <2 x i16> %{{.*}}, %{{.*}}

;; Test xor on vector:
define <2 x i16> @vector_xor(<2 x i16> %a, <2 x i16> %b) {
    %c = xor <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV:      BitwiseXor [[#VECTOR]] [[#C:]] [[#A]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = xor <2 x i16> %{{.*}}, %{{.*}}
