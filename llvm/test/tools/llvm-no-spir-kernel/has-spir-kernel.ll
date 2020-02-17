; RUN: not llvm-no-spir-kernel %s 2>&1 | FileCheck %s

; expected failure
; CHECK: error: Unexpected SPIR kernel occurrence: foo
define spir_kernel void @foo() {
bb:
  ret void
}
