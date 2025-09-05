; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_int4 -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability Int4TypeINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_int4"
; CHECK-SPIRV: TypeInt [[#Int4:]] 4 0
; CHECK-SPIRV: Constant [[#Int4]] [[#Const:]] 1
; CHECK-SPIRV: TypeFunction [[#]] [[#]] [[#Int4]]
; CHECK-SPIRV: TypePointer [[#Int3PtrTy:]] [[#]] [[#Int4]]
; CHECK-SPIRV: Variable [[#Int3PtrTy]] [[#Int3Ptr:]]
; CHECK-SPIRV: Store [[#Int3Ptr]] [[#Const]]
; CHECK-SPIRV: Load [[#Int4]] [[#Load:]] [[#Int3Ptr]]
; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[#]] [[#Load]]

; CHECK-LLVM: %[[#Alloc:]] = alloca i4, align 1
; CHECK-LLVM: store i4 1, ptr %[[#Alloc:]], align 1
; CHECK-LLVM: %[[#Load:]] = load i4, ptr %[[#Alloc]], align 1
; CHECK-LLVM: call spir_func void @boo(i4 %[[#Load]])


target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @foo() {
entry:
  %0 = alloca i4
  store i4 1, ptr %0
  %1 = load i4, ptr %0
  call spir_func void @boo(i4 %1)
  ret void
}

declare spir_func void @boo(i4)
