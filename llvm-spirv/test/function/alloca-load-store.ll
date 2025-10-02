; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"
 
; CHECK-SPIRV: TypeInt [[#I32:]] 32 0 
; CHECK-SPIRV: TypeFunction [[#FTY_BARFOO:]] [[#I32]] [[#I32]] 
; CHECK-SPIRV: TypePointer [[#PTR:]] [[#]] [[#I32]] 
; CHECK-SPIRV: TypePointer [[#PTR_FUNC:]] [[#]] [[#]] 
; CHECK-SPIRV: TypeFunction [[#FTY_GOO:]] [[#I32]] [[#I32]] [[#PTR_FUNC]] 

define i32 @bar(i32 %a) {
  %p = alloca i32
  store i32 %a, ptr %p
  %b = load i32, ptr %p
  ret i32 %b
}
; CHECK-SPIRV: Function [[#I32]] [[#]] 0 [[#FTY_BARFOO]] 
; CHECK-SPIRV: FunctionParameter [[#I32]] [[#BAR_ARG:]] 
; CHECK-SPIRV: Label [[#]] 
; CHECK-SPIRV: Variable [[#PTR]] [[#BAR_VAR:]] [[#]] 
; CHECK-SPIRV: Store [[#BAR_VAR]] [[#BAR_ARG]] 2 4 
; CHECK-SPIRV: Load [[#I32]] [[#BAR_LOAD:]] [[#BAR_VAR]] 2 4 
; CHECK-SPIRV: ReturnValue [[#BAR_LOAD]] 
; CHECK-SPIRV: FunctionEnd 

; CHECK-LLVM: define spir_func i32 @bar(i32 [[a:%.*]])
; CHECK-LLVM: [[p:%.*]] = alloca i32
; CHECK-LLVM: store i32 [[a]], ptr [[p]]
; CHECK-LLVM: [[b:%.*]] = load i32, ptr [[p]]

define i32 @foo(i32 %a) {
  %p = alloca i32
  store volatile i32 %a, ptr %p
  %b = load volatile i32, ptr %p
  ret i32 %b
}

; CHECK-SPIRV: Function [[#I32]] [[#]] 0 [[#FTY_BARFOO]] 
; CHECK-SPIRV: FunctionParameter [[#I32]] [[#FOO_ARG:]] 
; CHECK-SPIRV: Label [[#]] 
; CHECK-SPIRV: Variable [[#PTR]] [[#FOO_VAR:]] [[#]] 
; CHECK-SPIRV: Store [[#FOO_VAR]] [[#FOO_ARG]] 3 4 
; CHECK-SPIRV: Load [[#I32]] [[#FOO_LOAD:]] [[#FOO_VAR]] 3 4 
; CHECK-SPIRV: ReturnValue [[#FOO_LOAD]] 
; CHECK-SPIRV: FunctionEnd 

; CHECK-LLVM: define spir_func i32 @foo(i32 [[a:%.*]])
; CHECK-LLVM: [[p:%.*]] = alloca i32
; CHECK-LLVM: store volatile i32 [[a]], ptr [[p]]
; CHECK-LLVM: [[b:%.*]] = load volatile i32, ptr [[p]]

;; Test load and store in global address space.
define i32 @goo(i32 %a, ptr addrspace(1) %p) {
  store i32 %a, ptr addrspace(1) %p
  %b = load i32, ptr addrspace(1) %p
  ret i32 %b
}

; CHECK-SPIRV: Function [[#I32]] [[#]] 0 [[#FTY_GOO]] 
; CHECK-SPIRV: FunctionParameter [[#I32]] [[#GOO_ARG:]] 
; CHECK-SPIRV: FunctionParameter [[#PTR_FUNC]] [[#GOO_PTR:]] 
; CHECK-SPIRV: Label [[#]] 
; CHECK-SPIRV: Store [[#GOO_PTR]] [[#GOO_ARG]] 2 4 
; CHECK-SPIRV: Load [[#I32]] [[#GOO_LOAD:]] [[#GOO_PTR]] 2 4 
; CHECK-SPIRV: ReturnValue [[#GOO_LOAD]] 
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i32 @goo(i32 [[a:%.*]], ptr addrspace(1) [[p:%.*]])
; CHECK-LLVM: store i32 [[a]], ptr addrspace(1) [[p]]
; CHECK-LLVM: [[b:%.*]] = load i32, ptr addrspace(1) [[p]]
