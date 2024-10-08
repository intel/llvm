; XFAIL: *

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-ext=+SPV_INTEL_function_pointers -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -spirv-ext=+SPV_INTEL_function_pointers %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Check that aliases are dereferenced and translated to their aliasee values
; when used since they can't be translated directly.

; CHECK-SPIRV-DAG: Name [[#FOO:]] "foo"
; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#BAR:]] "bar"
; CHECK-SPIRV-DAG: Name [[#Y:]] "y"
; CHECK-SPIRV-DAG: Name [[#FOOPTR:]] "foo.alias"
; CHECK-SPIRV-DAG: Decorate [[#FOO]] LinkageAttributes "foo" Export
; CHECK-SPIRV-DAG: TypeInt [[#I32:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#I64:]] 64 0
; CHECK-SPIRV-DAG: TypeFunction [[#FOO_TYPE:]] [[#I32]] [[#I32]]
; CHECK-SPIRV-DAG: TypeVoid [[#VOID:]]
; CHECK-SPIRV-DAG: TypePointer [[#I64PTR:]] 7 [[#I64]]
; CHECK-SPIRV-DAG: TypeFunction [[#BAR_TYPE:]] [[#VOID]] [[#I64PTR]]
; CHECK-SPIRV-DAG: TypePointer [[#FOOPTR_TYPE:]] 7 [[#FOO_TYPE]]
; CHECK-SPIRV-DAG: ConstantFunctionPointerINTEL [[#FOOPTR_TYPE]] [[#FOOPTR]] [[#FOO]]

; CHECK-SPIRV: Function [[#I32]] [[#FOO]] 0 [[#FOO_TYPE]]

; CHECK-SPIRV: Function [[#VOID]] [[#BAR]] 0 [[#BAR_TYPE]]
; CHECK-SPIRV: FunctionParameter [[#I64PTR]] [[#Y]]
; CHECK-SPIRV: ConvertPtrToU [[#I64]] [[#PTRTOINT:]] [[#FOOPTR]]
; CHECK-SPIRV: Store [[#Y]] [[#PTRTOINT]] 2 8

; CHECK-LLVM: define spir_func i32 @foo(i32 %x)

; CHECK-LLVM: define spir_func void @bar(ptr %y)
; CHECK-LLVM: [[PTRTOINT:%.*]] = ptrtoint ptr @foo to i64
; CHECK-LLVM: store i64 [[PTRTOINT]], ptr %y, align 8

define spir_func i32 @foo(i32 %x) {
  ret i32 %x
}

@foo.alias = internal alias i32 (i32), ptr @foo

define spir_kernel void @bar(ptr %y) {
  store i64 ptrtoint (ptr @foo.alias to i64), ptr %y
  ret void
}
