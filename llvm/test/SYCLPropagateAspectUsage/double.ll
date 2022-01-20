; RUN: opt --PropagateAspectUsage < %s -S | FileCheck %s

; CHECK: dso_local spir_kernel void @kernel() !intel_used_aspects !0
define dso_local spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK: dso_local spir_func void @func() !intel_used_aspects !0
define dso_local spir_func void @func() {
  %tmp = alloca double
  ret void
}

; CHECK: !0 = !{i32 6}
