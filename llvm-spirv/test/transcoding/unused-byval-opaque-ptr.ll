; Ensure that a pointer passed by value is translated as a typed pointer even
; with the SPV_KHR_untyped_pointers extension enabled to preserve byval semantics.

; RUN: llvm-spirv %s -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -spirv-text -o %t.txt --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Name [[#Fun:]] "kernel"
; CHECK-SPIRV-DAG: Decorate [[#Param:]] FuncParamAttr 2
; CHECK-SPIRV-DAG: TypeStruct [[#StructTy:]]
; CHECK-SPIRV-DAG: TypePointer [[#PtrTy:]] [[#]] [[#StructTy]]
; CHECK-SPIRV: Function [[#]] [[#Fun]]
; CHECK-SPIRV: FunctionParameter [[#PtrTy]] [[#Param]]

; CHECK-LLVM: @kernel(ptr %arg0, ptr byval(%struct.Example) align 8 %arg1)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir-unknown-unknown"

%struct.Example = type { }

define spir_kernel void @kernel(ptr %arg0, ptr byval(%struct.Example) align 8 %arg1) {
entry:
  ret void
}
