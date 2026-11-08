; Ensure that sret pointer semantics is preserved when the parameter is unused
; (even with the SPV_KHR_untyped_pointers extension enabled).

; RUN: llvm-spirv %s -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -spirv-text -o %t.txt --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: FileCheck < %t.txt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED
; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers
; RUNx: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Name [[#FunBoo:]] "boo"
; CHECK-SPIRV-DAG: Name [[#FunBaz:]] "baz"
; CHECK-SPIRV-DAG: Decorate [[#ParamBoo:]] FuncParamAttr 3
; CHECK-SPIRV-DAG: Decorate [[#ParamBaz:]] FuncParamAttr 3
; CHECK-SPIRV-DAG: TypePointer [[#PtrTy7:]] 7 [[#StructTy:]]
; CHECK-SPIRV-DAG: TypePointer [[#PtrTy8:]] 8 [[#StructTy:]]
; CHECK-SPIRV-DAG: TypeStruct [[#StructTy]]

; CHECK-SPIRV-DAG: TypeFunction [[#BooTy:]] [[#]] [[#PtrTy7]] [[#]] {{$}}
; CHECK-SPIRV-DAG: TypeFunction [[#BazTy:]] [[#]] [[#PtrTy8]] {{$}}

; CHECK-SPIRV: Function [[#]] [[#FunBoo]] [[#]] [[#BooTy]]
; CHECK-SPIRV: FunctionParameter [[#PtrTy7]] [[#ParamBoo]]

; CHECK-SPIRV: Function [[#]] [[#FunBaz]] [[#]] [[#BazTy]]
; CHECK-SPIRV: FunctionParameter [[#PtrTy8]] [[#ParamBaz]]

; CHECK-SPIRV: FunctionParameter [[#PtrTy7]] [[#ParamBar:]]
; With untyped extension enabled addrspacecast is done to untyped pointer type in addrspace 8.
; CHECK-SPIRV: PtrCastToGeneric [[#]] [[#Cast:]] [[#ParamBar]]
; CHECK-SPIRV-TYPED: FunctionCall [[#]] [[#]] [[#FunBaz]] [[#Cast]]
; CHECK-SPIRV-UNTYPED: Bitcast [[#PtrTy8]] [[#CastBC:]] [[#Cast]]
; CHECK-SPIRV-UNTYPED: FunctionCall [[#]] [[#]] [[#FunBaz]] [[#CastBC]]

; CHECK-LLVM: call spir_func void @boo(ptr sret(%struct.Example) align 8
; Under untyped pointers the sret argument round-trips through the Writer's
; OpBitcast, so match either the addrspacecast result or the bitcast result.
; CHECK-LLVM: call spir_func void @baz(ptr addrspace(4) sret(%struct.Example) %{{.*}})

source_filename = "/app/example.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir-unknown-unknown"

%struct.Example = type { }

define spir_func i32 @foo() {
  %1 = alloca %struct.Example, align 8
  call void @boo(ptr sret(%struct.Example) align 8 %1, i32 noundef 42)
  ret i32 0
}

define spir_func void @bar(ptr sret(%struct.Example) %ret_ptr) {
  %cast = addrspacecast ptr %ret_ptr to ptr addrspace(4)
  call void @baz(ptr addrspace(4) sret(%struct.Example) %cast)
  ret void
}


declare void @boo(ptr sret(%struct.Example) align 8, i32 noundef)
declare void @baz(ptr addrspace(4) sret(%struct.Example))
