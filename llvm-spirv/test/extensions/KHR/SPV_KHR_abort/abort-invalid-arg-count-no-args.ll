; Negative test: OpAbortKHR takes exactly one Message operand; a call to
; `__spirv_AbortKHR` with no arguments must be rejected by the translator.

; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_KHR_abort -o /dev/null 2>&1 | FileCheck %s

; CHECK: InvalidInstruction
; CHECK: __spirv_AbortKHR must be called with exactly one Message argument

target triple = "spir64-unknown-unknown"

define spir_func void @abort_no_args() {
entry:
  call spir_func void @_Z16__spirv_AbortKHRv()
  ret void
}

declare spir_func void @_Z16__spirv_AbortKHRv()
