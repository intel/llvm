; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-SPIRV-DAG: Name [[#SCALAR_ADD:]] "scalar_add"
; CHECK-SPIRV-DAG: Name [[#SCALAR_SUB:]] "scalar_sub"
; CHECK-SPIRV-DAG: Name [[#SCALAR_MUL:]] "scalar_mul"
; CHECK-SPIRV-DAG: Name [[#SCALAR_UDIV:]] "scalar_udiv"
; CHECK-SPIRV-DAG: Name [[#SCALAR_SDIV:]] "scalar_sdiv"
; CHECK-SPIRV-DAG: Name [[#SCALAR_UREM:]] "scalar_urem"
; CHECK-SPIRV-DAG: Name [[#SCALAR_SREM:]] "scalar_srem"

;; TODO: add test for OpSNegate

; CHECK-SPIRV-NOT: DAG-FENCE

; CHECK-SPIRV-DAG: TypeInt [[#SCALAR:]] 32
; CHECK-SPIRV-DAG: TypeFunction [[#SCALAR_FN:]] [[#SCALAR]] [[#SCALAR]] [[#SCALAR]]

; CHECK-SPIRV-NOT: DAG-FENCE

define i32 @scalar_urem(i32 %a, i32 %b) {
    %c = urem i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      UMod [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_urem
; CHECK-LLVM: %[[R:.*]] = urem i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

define i32 @scalar_srem(i32 %a, i32 %b) {
    %c = srem i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      SRem [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_srem
; CHECK-LLVM: %[[R:.*]] = srem i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

define i32 @scalar_add(i32 %a, i32 %b) {
    %c = add i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      IAdd [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_add
; CHECK-LLVM: %[[R:.*]] = add i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

define i32 @scalar_sub(i32 %a, i32 %b) {
    %c = sub i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      ISub [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_sub
; CHECK-LLVM: %[[R:.*]] = sub i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

;; Test mul on scalar:
define i32 @scalar_mul(i32 %a, i32 %b) {
    %c = mul i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      IMul [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_mul
; CHECK-LLVM: %[[R:.*]] = mul i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

;; Test udiv on scalar:
define i32 @scalar_udiv(i32 %a, i32 %b) {
    %c = udiv i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      UDiv [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_udiv
; CHECK-LLVM: %[[R:.*]] = udiv i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

;; Test sdiv on scalar:
define i32 @scalar_sdiv(i32 %a, i32 %b) {
    %c = sdiv i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      SDiv [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_sdiv
; CHECK-LLVM: %[[R:.*]] = sdiv i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]
