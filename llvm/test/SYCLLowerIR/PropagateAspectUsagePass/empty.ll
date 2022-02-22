; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S | FileCheck %s --implicit-check-not "!intel_used_aspects"
;
; Test checks that no metadata propagates because MyStruct
; isn't used inside functions.

%MyStruct = type { i32 }

; CHECK: dso_local spir_kernel void @kernel() {
define dso_local spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK: dso_local spir_func void @func() {
define weak dso_local spir_func void @func() {
  ret void
}

!intel_types_that_use_aspects = !{!0}
!0 = !{!"MyStruct", i32 1}
