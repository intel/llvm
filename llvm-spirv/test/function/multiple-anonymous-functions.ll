; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: TypeInt [[#I32_TY:]] 32 0
; CHECK-SPIRV: TypeFunction [[#I32_FUNC_TY:]] [[#I32_TY]] [[#I32_TY]]
; CHECK-SPIRV: TypeFloat [[#F32_TY:]] 32
; CHECK-SPIRV: TypeFunction [[#F32_FUNC_TY:]] [[#F32_TY]] [[#F32_TY]]
; CHECK-SPIRV: Function [[#I32_TY]] [[#ID_1:]] 0 [[#I32_FUNC_TY]]
; CHECK-SPIRV: FunctionParameter [[#I32_TY]] [[#ID_A1:]]
; CHECK-SPIRV: Label [[#LBL_1:]]
; CHECK-SPIRV: ReturnValue [[#ID_A1]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define internal spir_func i32 @0(i32 %a)
; CHECK-LLVM:   ret i32 %a

define internal spir_func i32 @0(i32 %a) {
  ret i32 %a
}

; CHECK-SPIRV: Function [[#F32_TY]] [[#ID_2:]] 0 [[#F32_FUNC_TY]]
; CHECK-SPIRV: FunctionParameter [[#F32_TY]] [[#ID_A2:]]
; CHECK-SPIRV: Label [[#LBL_2:]]
; CHECK-SPIRV: ReturnValue [[#ID_A2]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define internal spir_func float @1(float %a)
; CHECK-LLVM:   ret float %a

define internal spir_func float @1(float %a) {
  ret float %a
}

; CHECK-SPIRV: FunctionCall [[#I32_TY]] [[#CALL1_ID:]] [[#ID_1]] [[#]]
; CHECK-SPIRV: FunctionCall [[#F32_TY]] [[#CALL2_ID:]] [[#ID_2]] [[#]]

; CHECK-LLVM: define spir_kernel void @foo(i32 [[a:%.*]])
; CHECK-LLVM:   [[call1:%.*]] = call spir_func i32 @0(i32 [[a]])
; CHECK-LLVM:   [[b:%.*]] = sitofp i32 [[a]] to float
; CHECK-LLVM:   [[call2:%.*]] = call spir_func float @1(float [[b]])

define spir_kernel void @foo(i32 %a) {
  %call1 = call spir_func i32 @0(i32 %a)
  %b = sitofp i32 %a to float
  %call2 = call spir_func float @1(float %b)
  ret void
}
