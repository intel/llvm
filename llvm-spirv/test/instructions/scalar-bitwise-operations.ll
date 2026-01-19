; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-SPIRV-DAG: Name [[#SCALAR_SHL:]] "scalar_shl"
; CHECK-SPIRV-DAG: Name [[#SCALAR_LSHR:]] "scalar_lshr"
; CHECK-SPIRV-DAG: Name [[#SCALAR_ASHR:]] "scalar_ashr"
; CHECK-SPIRV-DAG: Name [[#SCALAR_AND:]] "scalar_and"
; CHECK-SPIRV-DAG: Name [[#SCALAR_OR:]] "scalar_or"
; CHECK-SPIRV-DAG: Name [[#SCALAR_XOR:]] "scalar_xor"

; CHECK-SPIRV-NOT: DAG-FENCE

; CHECK-SPIRV-DAG: TypeInt [[#SCALAR:]] 32
; CHECK-SPIRV-DAG: TypeFunction [[#SCALAR_FN:]] [[#SCALAR]] [[#SCALAR]] [[#SCALAR]]

; CHECK-SPIRV-NOT: DAG-FENCE

;; Test shl on scalar:
define i32 @scalar_shl(i32 %a, i32 %b) {
    %c = shl i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      ShiftLeftLogical [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_shl
; CHECK-LLVM: %[[R:.*]] = shl i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

;; Test lshr on scalar:
define i32 @scalar_lshr(i32 %a, i32 %b) {
    %c = lshr i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      ShiftRightLogical [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_lshr
; CHECK-LLVM: %[[R:.*]] = lshr i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

;; Test ashr on scalar:
define i32 @scalar_ashr(i32 %a, i32 %b) {
    %c = ashr i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      ShiftRightArithmetic [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_ashr
; CHECK-LLVM: %[[R:.*]] = ashr i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

;; Test and on scalar:
define i32 @scalar_and(i32 %a, i32 %b) {
    %c = and i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      BitwiseAnd [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_and
; CHECK-LLVM: %[[R:.*]] = and i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

;; Test or on scalar:
define i32 @scalar_or(i32 %a, i32 %b) {
    %c = or i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      BitwiseOr [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_or
; CHECK-LLVM: %[[R:.*]] = or i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]

;; Test xor on scalar:
define i32 @scalar_xor(i32 %a, i32 %b) {
    %c = xor i32 %a, %b
    ret i32 %c
}

; CHECK-SPIRV:      Function [[#SCALAR]] [[#]] 0 [[#SCALAR_FN]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#A:]]
; CHECK-SPIRV: FunctionParameter [[#SCALAR]] [[#B:]]
; CHECK-SPIRV:      BitwiseXor [[#SCALAR]] [[#C:]] [[#A]] [[#B]]
; CHECK-SPIRV:      ReturnValue [[#C]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: @scalar_xor
; CHECK-LLVM: %[[R:.*]] = xor i32 %{{.*}}, %{{.*}}
; CHECK-LLVM: ret i32 %[[R]]
