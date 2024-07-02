; This test is intended to check that per-aspect device code split works as
; expected with SYCL_EXTERNAL functions

; RUN: sycl-post-link -properties -split=auto -symbols -S < %s -o %t.table
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
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix CHECK-M1-IR \
; RUN:     --implicit-check-not kernel0 --implicit-check-not bar

; RUN: sycl-module-split -split=auto -S < %s -o %t2
; RUN: FileCheck %s -input-file=%t2.table --check-prefix CHECK-TABLE
;
; RUN: FileCheck %s -input-file=%t2_0.sym --check-prefix CHECK-M0-SYMS \
; RUN:     --implicit-check-not foo --implicit-check-not kernel1
;
; RUN: FileCheck %s -input-file=%t2_1.sym --check-prefix CHECK-M1-SYMS \
; RUN:     --implicit-check-not foo --implicit-check-not kernel0
;
; RUN: FileCheck %s -input-file=%t2_2.sym --check-prefix CHECK-M2-SYMS \
; RUN:     --implicit-check-not kernel0 --implicit-check-not foo \
; RUN:     --implicit-check-not bar
;
; RUN: FileCheck %s -input-file=%t2_1.ll --check-prefix CHECK-M1-IR \
; RUN:     --implicit-check-not kernel0 --implicit-check-not bar

; We expect to see 3 modules generated:
;
; CHECK-TABLE: Code
; CHECK-TABLE-NEXT: _0.sym
; CHECK-TABLE-NEXT: _1.sym
; CHECK-TABLE-NEXT: _2.sym
; CHECK-TABLE-EMPTY:

; sycl-post-link aims to achieve two goals while doing splitting:
;   - each kernel must be self-contained, i.e. all functions called from a
;     kernel must reside in the same device image
;   - each entry point should be assigned to a correct device image in
;     accordance with selected device code split mode
;
; In this test @bar and @foo are SYCL_EXTERNAL functions and they are treated
; as entry points.
;
; @bar uses the same list of aspects as @kernel0 which calls it and therefore
; they can be put into the same device image. There also goes @baz, because of
; the same list of used aspects.
;
; CHECK-M0-SYMS: bar
; CHECK-M0-SYMS: baz
; CHECK-M0-SYMS: kernel0
;
; List of aspects used by @foo is different from the one attached to @kernel1
; which calls @foo (for example, @kernel1 uses an extra optional feature besides
; ones used in @foo). As a result, @foo should be both included into the same
; device image as @kernel1 to make it self contained, but at the same time it
; should also present in a separate device image, because it is an entry point
; with unique set of used aspects.
;
; CHECK-M1-SYMS: kernel1
;
; CHECK-M2-SYMS: foo
;
; @kernel1 uses @foo and therefore @foo should be present in the same module as
; @kernel1 as well
; CHECK-M1-IR-DAG: define spir_func void @foo
; CHECK-M1-IR-DAG: define spir_kernel void @kernel1


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

define spir_func void @foo() #0 !sycl_used_aspects !1 {
  ret void
}

define spir_func void @bar() #1 !sycl_used_aspects !2 {
  ret void
}

define spir_func void @baz() #1 !sycl_used_aspects !2 {
  ret void
}

define spir_kernel void @kernel0() #1 !sycl_used_aspects !2 {
entry:
  call void @bar()
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
