; RUN: opt --PropagateAspectUsage < %s -S | FileCheck %s
;
; Test checks that function graph is proceeded correctly.
;
;   K1  K2
;  /  \/  \
; F1  F2   F3
;
; F1 doesn't use optional type.
; F2 uses optional A.
; F3 uses optional B.

%A = type { i32 }

%B = type { i32 }

; CHECK-DAG: dso_local spir_kernel void @kernel1() !intel_used_aspects !2 {
define dso_local spir_kernel void @kernel1() {
  call spir_func void @func1()
  call spir_func void @func2()
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernel2() !intel_used_aspects !3 {
define dso_local spir_kernel void @kernel2() {
  call spir_func void @func2()
  call spir_func void @func3()
  ret void
}

; CHECK-DAG: dso_local spir_func void @func1() {
define dso_local spir_func void @func1() {
  %tmp = alloca i32
  ret void
}

; CHECK-DAG: dso_local spir_func void @func2() !intel_used_aspects !2 {
define dso_local spir_func void @func2() {
  %tmp = alloca %A
  ret void
}

; CHECK-DAG: dso_local spir_func void @func3() !intel_used_aspects !4 {
define dso_local spir_func void @func3() {
  %tmp = alloca %B
  ret void
}

!intel_types_that_use_aspects = !{!0, !1}
!0 = !{!"A", i32 1}
!1 = !{!"B", i32 2}

; CHECK-DAG: !2 = !{i32 1}
; CHECK-DAG: !3 = !{i32 1, i32 2}
; CHECK-DAG: !4 = !{i32 2}
