; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-SPIRV-DAG: Name [[#VECTOR_ADD:]] "vector_add"
; CHECK-SPIRV-DAG: Name [[#VECTOR_SUB:]] "vector_sub"
; CHECK-SPIRV-DAG: Name [[#VECTOR_MUL:]] "vector_mul"
; CHECK-SPIRV-DAG: Name [[#VECTOR_UDIV:]] "vector_udiv"
; CHECK-SPIRV-DAG: Name [[#VECTOR_SDIV:]] "vector_sdiv"
; CHECK-SPIRV-DAG: Name [[#VECTOR_UREM:]] "vector_urem"
; CHECK-SPIRV-DAG: Name [[#VECTOR_SREM:]] "vector_srem"

; CHECK-SPIRV-NOT: DAG-FENCE

; CHECK-SPIRV-DAG: TypeInt [[#I16:]] 16
; CHECK-SPIRV-DAG: TypeVector [[#VECTOR:]] [[#I16]]
; CHECK-SPIRV-DAG: TypeFunction [[#VECTOR_FN:]] [[#VECTOR]] [[#VECTOR]] [[#VECTOR]]

;; Test urem on vector:
define <2 x i16> @vector_urem(<2 x i16> %a, <2 x i16> %b) {
    %c = urem <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      UMod [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = urem <2 x i16> %{{.*}}, %{{.*}}

;; Test srem on vector:
define <2 x i16> @vector_srem(<2 x i16> %a, <2 x i16> %b) {
    %c = srem <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      SRem [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = srem <2 x i16> %{{.*}}, %{{.*}}

;; Test add on vector:
define <2 x i16> @vector_add(<2 x i16> %a, <2 x i16> %b) {
    %c = add <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      IAdd [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = add <2 x i16> %{{.*}}, %{{.*}}

;; Test sub on vector:
define <2 x i16> @vector_sub(<2 x i16> %a, <2 x i16> %b) {
    %c = sub <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      ISub [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = sub <2 x i16> %{{.*}}, %{{.*}}

;; Test mul on vector:
define <2 x i16> @vector_mul(<2 x i16> %a, <2 x i16> %b) {
    %c = mul <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      IMul [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = mul <2 x i16> %{{.*}}, %{{.*}}

;; Test udiv on vector:
define <2 x i16> @vector_udiv(<2 x i16> %a, <2 x i16> %b) {
    %c = udiv <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      UDiv [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = udiv <2 x i16> %{{.*}}, %{{.*}}

;; Test sdiv on vector:
define <2 x i16> @vector_sdiv(<2 x i16> %a, <2 x i16> %b) {
    %c = sdiv <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK-SPIRV:      Function [[#VECTOR]] [[#]] 0 [[#VECTOR_FN]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#VECTOR]] [[#B:]]
; CHECK-SPIRV:      SDiv [[#VECTOR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd
; CHECK-LLVM: %{{.*}} = sdiv <2 x i16> %{{.*}}, %{{.*}}
