; The test is intended to check that sycl-post-link correctly groups kernels
; by unique sets of aspects used in them

; RUN: sycl-post-link -properties -split=auto -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table --check-prefix CHECK-TABLE
;
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefix CHECK-M0-SYMS \
; RUN:     --implicit-check-not kernel3 --implicit-check-not kernel1 \
; RUN:     --implicit-check-not kernel2
;
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefix CHECK-M1-SYMS \
; RUN:     --implicit-check-not kernel0 --implicit-check-not kernel1 \
; RUN:     --implicit-check-not kernel2
;
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefix CHECK-M2-SYMS \
; RUN:     --implicit-check-not kernel0 --implicit-check-not kernel3

; RUN: sycl-module-split -split=auto -S < %s -o %t2
; RUN: FileCheck %s -input-file=%t2.table --check-prefix CHECK-TABLE
;
; RUN: FileCheck %s -input-file=%t2_0.sym --check-prefix CHECK-M0-SYMS \
; RUN:     --implicit-check-not kernel3 --implicit-check-not kernel1 \
; RUN:     --implicit-check-not kernel2
;
; RUN: FileCheck %s -input-file=%t2_1.sym --check-prefix CHECK-M1-SYMS \
; RUN:     --implicit-check-not kernel0 --implicit-check-not kernel1 \
; RUN:     --implicit-check-not kernel2
;
; RUN: FileCheck %s -input-file=%t2_2.sym --check-prefix CHECK-M2-SYMS \
; RUN:     --implicit-check-not kernel0 --implicit-check-not kernel3

; CHECK-TABLE: Code
; CHECK-TABLE-NEXT: _0.sym
; CHECK-TABLE-NEXT: _1.sym
; CHECK-TABLE-NEXT: _2.sym
; CHECK-TABLE-EMPTY:

; CHECK-M0-SYMS: kernel3

; CHECK-M1-SYMS: kernel1
; CHECK-M1-SYMS: kernel2

; CHECK-M2-SYMS: kernel0

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

define dso_local spir_kernel void @kernel0() #0 !sycl_used_aspects !1 {
entry:
  ret void
}

define dso_local spir_kernel void @kernel1() #0 !sycl_used_aspects !2 {
entry:
  ret void
}

define dso_local spir_kernel void @kernel2() #0 !sycl_used_aspects !3 {
entry:
  ret void
}

define dso_local spir_kernel void @kernel3() #0 !sycl_used_aspects !4 {
entry:
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }

!1 = !{i32 1}
!2 = !{i32 1, i32 2}
!3 = !{i32 2, i32 1}
!4 = !{i32 2, i32 3, i32 4}
