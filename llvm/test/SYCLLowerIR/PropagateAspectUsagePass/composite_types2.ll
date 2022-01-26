; RUN: opt --PropagateAspectUsage < %s -S | FileCheck %s
;
; Test checks simple composite structures.

%A = type { %B, %C }

%B = type { i32 }

%C = type { i32 }

; CHECK-DAG: dso_local spir_kernel void @kernelA() !intel_used_aspects !2 {
define dso_local spir_kernel void @kernelA() {
  %tmp = alloca %A
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernelB() !intel_used_aspects !3 {
define dso_local spir_kernel void @kernelB() {
  %tmp = alloca %B
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernelC() !intel_used_aspects !4 {
define dso_local spir_kernel void @kernelC() {
  %tmp = alloca %C
  ret void
}

!intel_types_that_use_aspects = !{!0, !1}
!0 = !{!"B", i32 1}
!1 = !{!"C", i32 2}

; Check metadata which should appear
; CHECK-DAG: !3 = !{i32 1}
; CHECK-DAG: !4 = !{i32 2}
; CHECK-DAG: !2 = !{i32 1, i32 2}
