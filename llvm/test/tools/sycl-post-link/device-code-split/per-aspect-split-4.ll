; This test is intended to check that we do not perform per-aspect split if
; it was disabled through one or another sycl-post-link option

; RUN: sycl-post-link -properties -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table --check-prefix CHECK-TABLE
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK-IR
;
; -lower-esimd is needed so sycl-post-link does not complain about no actions
; specified
; RUN: sycl-post-link -lower-esimd -ir-output-only -S < %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll --check-prefix CHECK-IR

; We expect to see only one module generated:
;
; CHECK-TABLE: Code
; CHECK-TABLE-NEXT: _0.ll
; CHECK-TABLE-EMPTY:

; Regardless of used aspects and sycl-module-id metadata, all kernel and
; functions should still be present.

; CHECK-IR-DAG: define spir_func void @foo
; CHECK-IR-DAG: define spir_func void @bar
; CHECK-IR-DAG: define spir_kernel void @kernel0
; CHECK-IR-DAG: define spir_kernel void @kernel1

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

define spir_func void @foo() #0 !sycl_used_aspects !1 {
  ret void
}

define spir_func void @bar() #1 !sycl_used_aspects !2 {
  ret void
}

define spir_kernel void @kernel0() #1 !sycl_used_aspects !2 {
entry:
  ret void
}

define spir_kernel void @kernel1() #0 !sycl_used_aspects !3 {
entry:
  call void @foo()
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }
attributes #1 = { "sycl-module-id"="TU2.cpp" }

!1 = !{i32 1}
!2 = !{i32 2}
!3 = !{i32 3, i32 1}

