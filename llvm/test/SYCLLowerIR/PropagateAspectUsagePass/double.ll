; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S | FileCheck %s
;
; Test checks that double's aspect is spotted and propagated.

; CHECK-DAG: dso_local spir_kernel void @kernel() !intel_used_aspects !0 {
define dso_local spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK-DAG: dso_local spir_func void @func() !intel_used_aspects !0 {
define dso_local spir_func void @func() {
  %tmp = alloca double
  ret void
}

; CHECK-DAG: !0 = !{i32 6}
