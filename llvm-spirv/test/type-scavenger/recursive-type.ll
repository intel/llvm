; Check that pointers whose types change are correctly handled by the
; translator.
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK: TypeInt [[I8:[0-9]+]] 8 0
; CHECK: TypePointer [[I8PTR:[0-9]+]] 7 [[I8]]
; CHECK: TypePointer [[I8PTRPTR:[0-9]+]] 7 [[I8PTR]]
; CHECK: TypePointer [[I8PTRPTRPTR:[0-9]+]] 7 [[I8PTRPTR]]

; Function Attrs: nounwind
define spir_kernel void @foo() {
; CHECK: Variable [[I8PTRPTR]] [[PTR:[0-9]+]] 7
; CHECK: Bitcast [[I8PTRPTRPTR]] [[STOREPTR:[0-9]+]] [[PTR]]
; CHECK: Store [[STOREPTR]] [[PTR]]
entry:
  %ptr = alloca ptr, align 4
  store ptr %ptr, ptr %ptr
  ret void
}
