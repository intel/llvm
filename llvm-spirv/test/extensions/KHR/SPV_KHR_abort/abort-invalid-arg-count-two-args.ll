; Negative test: OpAbortKHR takes exactly one Message operand; a call to
; `__spirv_AbortKHR` with two arguments must be rejected by the translator.

; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_KHR_abort -o /dev/null 2>&1 | FileCheck %s

; CHECK: InvalidInstruction
; CHECK: __spirv_AbortKHR must be called with exactly one Message argument

target triple = "spir64-unknown-unknown"

define spir_func void @abort_two_args(i32 %a, i32 %b) {
entry:
  call spir_func void @_Z16__spirv_AbortKHRii(i32 %a, i32 %b)
  ret void
}

declare spir_func void @_Z16__spirv_AbortKHRii(i32, i32)
