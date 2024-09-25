; Check whether the translator reports an error for a
; function other than printf with variadic arguments.

; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; CHECK: UnsupportedVarArgFunction: Variadic functions other than 'printf' are not supported in SPIR-V.
; Function Attrs: convergent
declare spir_func void @for__issue_diagnostic(i32 noundef, i32 noundef, ...) local_unnamed_addr

define i32 @foo() nounwind {
  call spir_func void (i32, i32, ...) @for__issue_diagnostic(i32 noundef 41, i32 noundef 0)
  ret i32 0
}
