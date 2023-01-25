; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
;
; Test checks that double's aspect is spotted and propagated.

%composite = type { double }

; CHECK: spir_kernel void @kernel() !sycl_used_aspects ![[MDID:[0-9]+]]
define spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK: spir_func void @func() !sycl_used_aspects ![[MDID]] {
define spir_func void @func() {
  %tmp = alloca double
  ret void
}

; CHECK: spir_func void @func.array() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.array() {
  %tmp = alloca [4 x double]
  ret void
}

; CHECK: spir_func void @func.vector() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.vector() {
  %tmp = alloca <4 x double>
  ret void
}

; CHECK: spir_func void @func.composite() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.composite() {
  %tmp = alloca %composite
  ret void
}

!sycl_aspects = !{!0}
!0 = !{!"fp64", i32 6}

; CHECK: ![[MDID]] = !{i32 6}
