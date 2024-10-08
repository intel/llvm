; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Name [[#Fun:]] "_Z3booi"
; CHECK-SPIRV-DAG: Decorate [[#Param:]] FuncParamAttr 3
; CHECK-SPIRV-DAG: TypePointer [[#PtrTy:]] [[#]] [[#StructTy:]]
; CHECK-SPIRV-DAG: TypeStruct [[#StructTy]]
; CHECK-SPIRV: Function [[#]] [[#Fun]]
; CHECK-SPIRV: FunctionParameter [[#PtrTy:]] [[#Param]]

; CHECK-LLVM: call spir_func void @_Z3booi(ptr sret(%struct.Example) align 8

source_filename = "/app/example.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir-unknown-unknown"

%struct.Example = type { }

define spir_func i32 @foo() {
  %1 = alloca %struct.Example, align 8
  call void @_Z3booi(ptr sret(%struct.Example) align 8 %1, i32 noundef 42)
  ret i32 0
}

declare void @_Z3booi(ptr sret(%struct.Example) align 8, i32 noundef)
