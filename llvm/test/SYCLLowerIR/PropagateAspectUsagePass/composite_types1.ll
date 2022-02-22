; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S | FileCheck %s
;
; Test checks simple composite structures.

; Optional
%A = type { i32 }

; Not optional
%B = type { i32 }

; Not optional
%C = type { i32 }

; Contains optional - %A
%D1 = type { %A, %B, %C }

; Doesn't contain optionals
%D2 = type { %B, %C }

; Contains %D1 which contains optional %A
%E = type { %B, %C, %D1 }

; Contains a pointer on %D1 which contains optional %A
%F1 = type { %B, %C*, %D1* }

; Contains a pointer on %D2 which doesn't contain optionals
%F2 = type { %B, %C*, %D2* }

; CHECK-DAG: dso_local spir_kernel void @kernelD1() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernelD1() {
  call spir_func void @funcD1()
  ret void
}

; CHECK-DAG: dso_local spir_func void @funcD1() !intel_used_aspects !1 {
define dso_local spir_func void @funcD1() {
  %tmp = alloca %D1
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernelD2() {
define dso_local spir_kernel void @kernelD2() {
  call spir_func void @funcD2()
  ret void
}

; CHECK-DAG: dso_local spir_func void @funcD2() {
define dso_local spir_func void @funcD2() {
  %tmp = alloca %D2
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernelE() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernelE() {
  call spir_func void @funcE()
  ret void
}

; CHECK-DAG: dso_local spir_func void @funcE() !intel_used_aspects !1 {
define dso_local spir_func void @funcE() {
  %tmp = alloca %E
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernelF1() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernelF1() {
  call spir_func void @funcF1()
  ret void
}

; CHECK-DAG: dso_local spir_func void @funcF1() !intel_used_aspects !1 {
define dso_local spir_func void @funcF1() {
  %tmp = alloca %F1
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernelF2() {
define dso_local spir_kernel void @kernelF2() {
  call spir_func void @funcF2()
  ret void
}

; CHECK-DAG: dso_local spir_func void @funcF2() {
define dso_local spir_func void @funcF2() {
  %tmp = alloca %F2
  ret void
}

!intel_types_that_use_aspects = !{!0}
!0 = !{!"A", i32 1}

; CHECK-DAG: !1 = !{i32 1}

