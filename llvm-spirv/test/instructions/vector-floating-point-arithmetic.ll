; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-SPIRV-DAG: Name [[#VECTOR_FNEG:]] "vector_fneg"
; CHECK-SPIRV-DAG: Name [[#VECTOR_FADD:]] "vector_fadd"
; CHECK-SPIRV-DAG: Name [[#VECTOR_FSUB:]] "vector_fsub"
; CHECK-SPIRV-DAG: Name [[#VECTOR_FMUL:]] "vector_fmul"
; CHECK-SPIRV-DAG: Name [[#VECTOR_FDIV:]] "vector_fdiv"
; CHECK-SPIRV-DAG: Name [[#VECTOR_FREM:]] "vector_frem"

; CHECK-SPIRV-NOT: DAG-FENCE

; CHECK-SPIRV-DAG: TypeFloat [[#FP16:]] 16
; CHECK-SPIRV-DAG: TypeVector [[#VECTOR:]] [[#FP16]]
; CHECK-SPIRV-DAG: TypeFunction [[#VECTOR_FN:]] [[#VECTOR]] [[#VECTOR]] [[#VECTOR]]

;; Test fneg on vector:
define <2 x half> @vector_fneg(<2 x half> %a, <2 x half> %unused) {
    %c = fneg <2 x half> %a
    ret <2 x half> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV:      FNegate [[#VECTOR]] [[#C:]] [[#A]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = fneg <2 x half> %{{.*}}

;; Test fadd on vector:
define <2 x half> @vector_fadd(<2 x half> %a, <2 x half> %b) {
    %c = fadd <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      FAdd [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = fadd <2 x half> %{{.*}}, %{{.*}}

;; Test fsub on vector:
define <2 x half> @vector_fsub(<2 x half> %a, <2 x half> %b) {
    %c = fsub <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      FSub [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = fsub <2 x half> %{{.*}}, %{{.*}}

;; Test fmul on vector:
define <2 x half> @vector_fmul(<2 x half> %a, <2 x half> %b) {
    %c = fmul <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      FMul [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = fmul <2 x half> %{{.*}}, %{{.*}}

;; Test fdiv on vector:
define <2 x half> @vector_fdiv(<2 x half> %a, <2 x half> %b) {
    %c = fdiv <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      FDiv [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = fdiv <2 x half> %{{.*}}, %{{.*}}

;; Test frem on vector:
define <2 x half> @vector_frem(<2 x half> %a, <2 x half> %b) {
    %c = frem <2 x half> %a, %b
    ret <2 x half> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      FRem [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = frem <2 x half> %{{.*}}, %{{.*}}
