; RUN: not llvm-no-spir-kernel  %s 
; expected failure

define void @foo() {
bb:
  ret void
}

; expected failure
define spir_kernel void @foo2() {
bb:
  ret void
}

