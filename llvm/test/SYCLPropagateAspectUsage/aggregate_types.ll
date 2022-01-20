; RUN: opt --PropagateAspectUsage < %s -S
; | FileCheck %s
;
; Test checks that no metadata propagates because MyStruct
; isn't used inside functions.

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

; Contains %D which contains optional %A
%E = type { %B, %C, %D1 }

; Contains a pointer on %D which contains optional %A
%F = type { %B, %C*, %D1* }

; CHECK: dso_local spir_kernel void @kernel() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernelD1() {
  call spir_func void @funcD1()
  ret void
}

define dso_local spir_func void @funcD1() {
  %tmp = alloca %D1
  ret void
}

define dso_local spir_kernel void @kernelD2() {
  call spir_func void @funcD2()
  ret void
}

define dso_local spir_func void @funcD2() {
  %tmp = alloca %D2
  ret void
}

define dso_local spir_kernel void @kernelE() {
  call spir_func void @funcE()
  ret void
}

define dso_local spir_func void @funcE() {
  %tmp = alloca %E
  ret void
}

define dso_local spir_kernel void @kernelF() {
  call spir_func void @funcF()
  ret void
}

define dso_local spir_func void @funcF() {
  %tmp = alloca %F
  ret void
}

!intel_types_that_use_aspects = !{!0}
!0 = !{!"A", i32 1}

