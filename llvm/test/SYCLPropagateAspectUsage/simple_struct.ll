; RUN: opt --PropagateAspectUsage < %s -S | FileCheck %s
;
; Test checks that no metadata propagates because MyStruct
; isn't used inside functions.

%MyStruct = type { i32 }

; CHECK: dso_local spir_kernel void @kernel() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK: dso_local spir_func void @func() !intel_used_aspects !1 {
define weak dso_local spir_func void @func() {
  %struct = alloca %MyStruct
  ret void
}

!intel_types_that_use_aspects = !{!0}
!0 = !{!"MyStruct", i32 8}

; CHECK: !1 = !{i32 8}
