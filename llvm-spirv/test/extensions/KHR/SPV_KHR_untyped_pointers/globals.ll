; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spv
; TODO: enable back once spirv-tools are updated.
; R/UN: spirv-val %t.spv
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV-DAG: TypeInt [[#I16:]] 16 0
; CHECK-SPIRV-DAG: Constant [[#I16]] [[#CONST0:]] 0
; CHECK-SPIRV-DAG: TypeInt [[#I32:]] 32 0
; CHECK-SPIRV-DAG: TypeUntypedPointerKHR [[#PTRTY:]] 5
; CHECK-SPIRV-DAG: TypeUntypedPointerKHR [[#LOCALPTRTY:]] 4

; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#VARA:]] 5 [[#I16]] [[#CONST0]]
; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#VARB:]] 5 [[#I32]]
; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#VARC:]] 5 [[#PTRTY]] [[#VARA]]
; CHECK-SPIRV: UntypedVariableKHR [[#LOCALPTRTY]] [[#VARD:]] 4 [[#PTRTY]]

; CHECK-LLVM: @a = addrspace(1) global i16 0
; CHECK-LLVM: @b = external addrspace(1) global i32
; CHECK-LLVM: @c = addrspace(1) global ptr addrspace(1) @a
; CHECK-LLVM: @d = external addrspace(3) global ptr addrspace(1)

@a = addrspace(1) global i16 0
@b = external addrspace(1) global i32
@c = addrspace(1) global ptr addrspace(1) @a
@d = external addrspace(3) global ptr addrspace(1)

; Function Attrs: nounwind
define spir_func void @foo() {
entry:
  ret void
}
