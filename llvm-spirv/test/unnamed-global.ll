; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: TypeInt [[#INT_ID:]] 8 0
; CHECK: Constant [[#INT_ID]] [[#CONST_ID:]] 123
; CHECK: TypePointer [[#PTR_ID:]] [[#SC:]] [[#INT_ID]]
; CHECK: Variable [[#PTR_ID]] [[#VAR_ID:]] [[#SC]] [[#CONST_ID]]

; CHECK-LLVM: @0 = addrspace(1) global i8 123

@0 = addrspace(1) global i8 123

; Function Attrs: nounwind
define spir_kernel void @foo() #0 {
  ret void
}

attributes #0 = { nounwind }
