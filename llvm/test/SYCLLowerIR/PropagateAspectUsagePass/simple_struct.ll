; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S | FileCheck %s
;
; Test checks usage of simple struct.

%MyStruct = type { i32 }

; CHECK-DAG: dso_local spir_kernel void @kernel() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK-DAG: dso_local spir_func void @func() !intel_used_aspects !1 {
define weak dso_local spir_func void @func() {
  %struct = alloca %MyStruct
  ret void
}

!intel_types_that_use_aspects = !{!0}
!0 = !{!"MyStruct", i32 8}

; CHECK-DAG: !1 = !{i32 8}
