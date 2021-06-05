; OpenCL C source:
; char foo(char a) {
;     return a;
; }
; void bar() {
;    int (*fun_ptr)(int) = &foo;
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeFunction [[#FOO_TY:]] [[#]] [[#]]
; CHECK-SPIRV: TypeFunction [[#DEST_TY:]] [[#]] [[#]]
; CHECK-SPIRV: TypePointer [[#DEST_TY_PTR:]] [[#]] [[#DEST_TY]]
; CHECK-SPIRV: TypePointer [[#FOO_TY_PTR:]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV: ConstFunctionPointerINTEL [[#FOO_TY_PTR]] [[#FOO_PTR:]] [[#FOO:]]
; CHECK-SPIRV: Function [[#]] [[#FOO]] [[#]] [[#FOO_TY]]

; CHECK-SPIRV: Bitcast [[#DEST_TY_PTR]] [[#]] [[#FOO_PTR]]

; CHECK-LLVM: bitcast i8 (i8)* @foo to i32 (i32)*

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
  %1 = alloca i32 (i32)*, align 4
  store i32 (i32)* bitcast (i8 (i8)* @foo to i32 (i32)*), i32 (i32)** %1, align 4
  ret void
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 0e1accd0f726eef2c47be9f37dd0a06cb50d207e)"}
