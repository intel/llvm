; This test checks that the post-link tool produces a correct resulting file
; table and a symbol file for an input module with two kernels when no code
; splitting is requested.
;
; RUN: sycl-post-link -properties -symbols -spec-const=native -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files.table --check-prefixes CHECK-TABLE
; RUN: FileCheck %s -input-file=%t.files_0.sym --match-full-lines --check-prefixes CHECK-SYM

define dso_local spir_kernel void @KERNEL_AAA() {
; CHECK-SYM-NOT: {{[a-zA-Z0-9._@]+}}
; CHECK-SYM: KERNEL_AAA
entry:
  ret void
}

define dso_local spir_kernel void @KERNEL_BBB() {
; CHECK-SYM-NEXT: KERNEL_BBB
; CHECK-SYM-EMPTY:
entry:
  ret void
}

; CHECK-TABLE: [Code|Properties|Symbols]
; CHECK-TABLE-NEXT: {{.*}}files_0.sym
; CHECK-TABLE-EMPTY:
