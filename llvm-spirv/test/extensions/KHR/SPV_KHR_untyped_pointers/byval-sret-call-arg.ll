; Check that byval/sret arguments at call site are also emitted as typed
; pointers, to match function parameter.

; RUN: llvm-spirv %s -o %t.spt -spirv-text --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Name [[#Callee:]] "callee"
; CHECK-SPIRV-DAG: TypeStruct [[#StructTy:]] [[#]] [[#]]
; CHECK-SPIRV-DAG: TypePointer [[#TypedPtrTy:]] 7 [[#StructTy]]
; CHECK-SPIRV: FunctionParameter [[#TypedPtrTy]] [[#]]
; CHECK-SPIRV: FunctionParameter [[#TypedPtrTy]] [[#]]
; CHECK-SPIRV: Bitcast [[#TypedPtrTy]] [[#SretArg:]] [[#]]
; CHECK-SPIRV: Bitcast [[#TypedPtrTy]] [[#ByvalArg:]] [[#]]
; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[#Callee]] [[#SretArg]] [[#ByvalArg]]

; CHECK-LLVM: define spir_func void @callee(ptr sret(%struct.S) %r, ptr byval(%struct.S) %p)
; CHECK-LLVM: %[[R:.*]] = alloca %struct.S
; CHECK-LLVM: %[[TMP:.*]] = alloca %struct.S
; CHECK-LLVM: %[[SRET:.*]] = bitcast ptr %[[R]] to ptr
; CHECK-LLVM: %[[BYVAL:.*]] = bitcast ptr %[[TMP]] to ptr
; CHECK-LLVM: call spir_func void @callee(ptr sret(%struct.S) %[[SRET]], ptr byval(%struct.S) %[[BYVAL]])

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir-unknown-unknown"

%struct.S = type { i32, float }

define spir_func void @callee(ptr sret(%struct.S) %r, ptr byval(%struct.S) %p) {
entry:
  ret void
}

define spir_kernel void @kern() {
entry:
  %r = alloca %struct.S
  %tmp = alloca %struct.S
  call spir_func void @callee(ptr sret(%struct.S) %r, ptr byval(%struct.S) %tmp)
  ret void
}
