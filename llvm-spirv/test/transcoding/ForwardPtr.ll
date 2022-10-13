; RUN: llvm-as < %s | llvm-spirv -spirv-ext=+all -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeForwardPointer [[#FwdPtr:]] 8
; CHECK-SPIRV: TypeStruct [[#FuncArg:]] [[#FwdPtr]]
; CHECK-SPIRV: TypeStruct [[#ArgsSec:]] [[#FwdPtr]]
; CHECK-SPIRV: TypeStruct [[#A:]] [[#]] [[#ArgsSec:]]
; CHECK-SPIRV: TypePointer [[#FwdPtr]] 8 [[#A]]
; CHECK-SPIRV: TypeFunction [[#]] [[#]] [[#FwdPtr]]

; CHECK-LLVM: %struct.FuncArg = type { %class.A addrspace(4)* }
; CHECK-LLVM: %class.A = type { %class.ArgFirst, %structArgSec }
; CHECK-LLVM: %class.ArgFirst = type { void (%class.A addrspace(4)*)* }
; CHECK-LLVM: %structArgSec = type { %class.A addrspace(4)* }

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.FuncArg = type { %class.A addrspace(4)* }
%class.A = type { %class.ArgFirst, %structArgSec }
%class.ArgFirst = type { void (%class.A addrspace(4)*)* }
%structArgSec = type { %class.A addrspace(4)* }

declare spir_func i1 @Caller(%struct.FuncArg addrspace(4)* ) align 2

define spir_func void @MainFunc(%struct.FuncArg addrspace(4)* %context) {
  entry:
  %call = call spir_func zeroext i1 @Caller( %struct.FuncArg addrspace(4)* undef) 
  ret void
}

