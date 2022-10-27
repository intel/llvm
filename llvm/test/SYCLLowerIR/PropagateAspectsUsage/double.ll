; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
;
; Test checks that double's aspect is spotted and propagated.

%composite = type { double }

; CHECK: spir_kernel void @kernel() !intel_used_aspects !0 {
define spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK: spir_func void @func() !intel_used_aspects !0 {
define spir_func void @func() {
  %tmp = alloca double
  ret void
}

; CHECK: spir_func void @func.array() !intel_used_aspects !0 {
define spir_func void @func.array() {
  %tmp = alloca [4 x double]
  ret void
}

; CHECK: spir_func void @func.vector() !intel_used_aspects !0 {
define spir_func void @func.vector() {
  %tmp = alloca <4 x double>
  ret void
}

; CHECK: spir_func void @func.composite() !intel_used_aspects !0 {
define spir_func void @func.composite() {
  %tmp = alloca %composite
  ret void
}

; CHECK: !0 = !{i32 6}
