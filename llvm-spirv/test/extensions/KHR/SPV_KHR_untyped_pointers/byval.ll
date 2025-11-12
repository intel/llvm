; Ensure that a typed pointer passed by value is converted to an untyped pointer prior usage.

; RUN: llvm-spirv %s -spirv-text -o %t.spt --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Name [[#Fun:]] "kernel"
; CHECK-SPIRV-DAG: Decorate [[#Param:]] FuncParamAttr 2
; CHECK-SPIRV-DAG: TypeUntypedPointerKHR [[#UntypedPtrTy:]] 7
; CHECK-SPIRV-DAG: TypeStruct [[#StructTy:]]
; CHECK-SPIRV-DAG: TypePointer [[#PtrTy:]] 7 [[#StructTy]]
; CHECK-SPIRV-DAG: TypeInt [[#I32Ty:]] 32 0

; CHECK-SPIRV: Function [[#]] [[#Fun]]
; CHECK-SPIRV: FunctionParameter [[#PtrTy]] [[#Param]]

; CHECK-SPIRV: Bitcast [[#UntypedPtrTy]] [[#BC:]] [[#Param]]
; CHECK-SPIRV: Load [[#I32Ty]] [[#]] [[#BC]]

; CHECK-LLVM: @kernel(ptr %arg0, ptr byval(%struct.Example) align 8 %arg1)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir-unknown-unknown"

%struct.Example = type { }

define spir_kernel void @kernel(ptr %arg0, ptr byval(%struct.Example) align 8 %arg1) {
entry:
  %0 = load i32, ptr %arg1, align 8
  ret void
}
