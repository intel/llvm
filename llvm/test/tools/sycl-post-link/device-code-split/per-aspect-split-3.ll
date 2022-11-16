; This test is intended to check that per-aspect device code split works as
; expected with SYCL_EXTERNAL functions

; RUN: sycl-post-link -split=auto -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table --check-prefix CHECK-TABLE
;
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefix CHECK-M0-SYMS \
; RUN:     --implicit-check-not foo --implicit-check-not kernel1
;
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefix CHECK-M1-SYMS \
; RUN:     --implicit-check-not foo --implicit-check-not kernel0
;
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefix CHECK-M2-SYMS \
; RUN:     --implicit-check-not kernel0 --implicit-check-not foo \
; RUN:     --implicit-check-not bar
;
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefix CHECK-M2-IR \
; RUN:     --implicit-check-not kernel0 --implicit-check-not bar

; We expect to see 3 modules generated:
;
; CHECK-TABLE: Code
; CHECK-TABLE-NEXT: _0.sym
; CHECK-TABLE-NEXT: _1.sym
; CHECK-TABLE-NEXT: _2.sym
; CHECK-TABLE-EMPTY:

; Both @bar and @kernel0 use the same aspects and contained within the same
; translation unit, so they should be bundled together.
;
; CHECK-M0-SYMS: bar
; CHECK-M0-SYMS: kernel0

; @foo is a SYCL_EXTERNAL function, it should be exported and outlined into a
; separate translation unit because of used aspects
;
; CHECK-M1-SYMS: foo

; CHECK-M2-SYMS: kernel1
;
; @kernel1 uses @foo and therefore @foo should be present in the same module as
; @kernel1 as well
; CHECK-M2-IR-DAG: define spir_func void @foo
; CHECK-M2-IR-DAG: define spir_kernel void @kernel1


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
