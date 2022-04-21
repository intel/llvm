; UNSUPPORTED: system-windows

; Check the return code
; RUN: llvm-no-spir-kernel %s; \
; RUN: if [ $? = 1 ]; then exit 0; else exit 1; fi

; expected failure
define spir_kernel void @foo() {
bb:
  ret void
}
