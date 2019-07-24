; RUN: not llvm-no-spir-kernel  %s 

; expected failure
define spir_kernel void @foo() {
bb:
  ret void
}


