; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S | FileCheck %s
;
; Test checks that !intel_used_aspects metadata appears on both kernels and
; SYCL_EXTERNAL functions

%A = type { i32 }

; CHECK-DAG: dso_local spir_kernel void @kernel() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel() {
  call spir_func void @foo()
  ret void
}

; CHECK-DAG: dso_local spir_func void @func() #0 !intel_used_aspects !1 {
define weak dso_local spir_func void @func() #0 {
  call spir_func void @foo()
  ret void
}

define spir_func void @foo() {
  %tmp = alloca %A
  ret void
}

attributes #0 = {"sycl-module-id"="module-1"}

!intel_types_that_use_aspects = !{!0}
!0 = !{!"A", i32 1}

; CHECK-DAG: !1 = !{i32 1}
