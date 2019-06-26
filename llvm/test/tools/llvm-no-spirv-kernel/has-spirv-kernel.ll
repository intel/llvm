; RUN: not llvm-no-spirv-kernel  %s 

; expected failure
define spir_kernel void @foo() {
bb:
  ret void
}


