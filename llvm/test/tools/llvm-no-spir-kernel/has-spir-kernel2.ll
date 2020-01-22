; RUN: not llvm-no-spir-kernel %s 2>&1 | FileCheck %s

; expected no failures
define void @foo() {
bb:
  ret void
}

; expected failure
; CHECK: error: Unexpected SPIR kernel occurrence:
; CHECK-SAME: foo2
define spir_kernel void @foo2() {
bb:
  ret void
}

; Check the return code
; RUN: llvm-no-spir-kernel %s; \
; RUN: if [ $? = 1 ]; then exit 0; else exit 1; fi
