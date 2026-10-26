; OpenCL C source:
; char foo(char a) {
;     return a;
; }
; void bar() {
;    int (*fun_ptr)(int) = &foo;
;    fun_ptr(0);
; }

; RUN: llvm-spirv %s -spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv %s -spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -o %t.u.spv
; RUN: llvm-spirv %t.u.spv -to-text -o %t.u.spt
; RUN: FileCheck < %t.u.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED
; RUN: llvm-spirv -r %t.u.spv -o %t.ru.bc
; RUN: llvm-dis %t.ru.bc -o %t.ru.ll
; RUN: FileCheck < %t.ru.ll %s --check-prefix=CHECK-LLVM

; TODO: reader currently crashes with output of llc for this test with function pointers (for both typed and untyped pointers)
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -filetype=obj %s -o %t.llc.spv %}
; RUNx: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -filetype=obj %s -o %t.llc.u.spv %}
; RUNx: %if spirv-backend %{ llvm-spirv -r %t.llc.u.spv -o %t.llc.u.rev.bc %}

; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV-UNTYPED: Extension "SPV_KHR_untyped_pointers"
; CHECK-SPIRV-DAG: TypeInt [[#I8:]] 8
; CHECK-SPIRV-DAG: TypeInt [[#I32:]] 32
; CHECK-SPIRV-DAG: TypeFunction [[#FOO_TY:]] [[#I8]] [[#I8]]
; CHECK-SPIRV-DAG: TypeFunction [[#DEST_TY:]] [[#I32]] [[#I32]]
; CHECK-SPIRV-TYPED-DAG: TypePointer [[#DEST_TY_PTR:]] [[#]] [[#DEST_TY]]
; CHECK-SPIRV-TYPED-DAG: TypePointer [[#FOO_TY_PTR:]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV-UNTYPED-DAG: TypeUntypedPointerKHR [[#FOO_TY_PTR:]] 7
; CHECK-SPIRV: ConstantFunctionPointerINTEL [[#FOO_TY_PTR]] [[#FOO_PTR:]] [[#FOO:]]
; CHECK-SPIRV: Function [[#]] [[#FOO]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV-TYPED: Bitcast [[#DEST_TY_PTR]] [[#]] [[#FOO_PTR]]
; CHECK-SPIRV-UNTYPED: Bitcast [[#FOO_TY_PTR]] [[#]] [[#FOO_PTR]]

; CHECK-LLVM: bitcast ptr @foo to ptr

; ModuleID = './example.c'
source_filename = "./example.c"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: noinline nounwind optnone
define dso_local spir_func signext i8 @foo(i8 signext %0) #0 {
  ret i8 %0
}

; Function Attrs: noinline nounwind optnone
define dso_local spir_func void @bar() #0 {
  %1 = call i32 @foo(i32 0)
  ret void
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 0e1accd0f726eef2c47be9f37dd0a06cb50d207e)"}
