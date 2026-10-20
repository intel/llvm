; RUN: llvm-as < %s | llvm-spirv -spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv %t.spv -spirv-ext=+SPV_INTEL_function_pointers -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-as < %s | llvm-spirv -spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -o %t.u.spv
; RUN: llvm-spirv %t.u.spv -spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -to-text -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED
; RUN: llvm-spirv -r %t.u.spv -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM < %t.llc.rev.ll %}

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -filetype=obj %s -o %t.llc.u.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.u.spv -o %t.llc.u.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.u.rev.bc -o %t.llc.u.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM < %t.llc.u.rev.ll %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"


; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV-UNTYPED: Extension "SPV_KHR_untyped_pointers"
; CHECK-SPIRV-TYPED: TypeFunction [[#FOO_TY:]] [[#]] [[#]]
; CHECK-SPIRV-TYPED: TypePointer [[#FOO_TY_PTR:]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV-UNTYPED: TypeUntypedPointerKHR [[#FOO_TY_PTR:]] [[#]]
; CHECK-SPIRV-UNTYPED: TypeFunction [[#FOO_TY:]] [[#]] [[#]]
; CHECK-SPIRV: ConstantFunctionPointerINTEL [[#FOO_TY_PTR]] [[#FOO_PTR:]] [[#FOO:]]
; CHECK-SPIRV: Function [[#]] [[#]] [[#]] [[#FOO_TY]]

; CHECK-LLVM: @two = internal addrspace(1) global ptr @_Z4barrii
; CHECK-LLVM: define spir_func i32 @_Z4barrii(i32 %[[#]], i32 %[[#]])

@two = internal addrspace(1) global ptr @_Z4barrii, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define protected spir_func noundef i32 @_Z4barrii(i32 %0, i32 %1) {
entry:
  ret i32 1
}
