; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S > %t.ll
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-FIRST
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-SECOND
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-THIRD
;
; Test checks that function graph is proceeded correctly.
;
;   K
;  /  \
; F1  F2
;  \  / \
;   F3   F4
;
; F3 uses optional A.
; F4 uses optional B.

%A = type { i32 }

%B = type { i32 }

; CHECK-FIRST-DAG: dso_local spir_kernel void @kernel() !intel_used_aspects ![[NODE1:[0-9]+]] {
define dso_local spir_kernel void @kernel() {
  call spir_func void @func1()
  call spir_func void @func2()
  ret void
}

; CHECK-SECOND-DAG: dso_local spir_func void @func1() !intel_used_aspects ![[NODE2:[0-9]+]] {
define dso_local spir_func void @func1() {
  call spir_func void @func3()
  ret void
}

; CHECK-FIRST-DAG: dso_local spir_func void @func2() !intel_used_aspects ![[NODE1:[0-9]+]] {
define dso_local spir_func void @func2() {
  call spir_func void @func3()
  call spir_func void @func4()
  ret void
}

; CHECK-SECOND-DAG: dso_local spir_func void @func3() !intel_used_aspects ![[NODE2:[0-9]+]] {
define dso_local spir_func void @func3() {
  %tmp = alloca %A
  ret void
}

; CHECK-THIRD-DAG: dso_local spir_func void @func4() !intel_used_aspects ![[NODE3:[0-9]+]] {
define dso_local spir_func void @func4() {
  %tmp = alloca %B
  ret void
}

!intel_types_that_use_aspects = !{!0, !1}
!0 = !{!"A", i32 1}
!1 = !{!"B", i32 2}

; CHECK-FIRST: ![[NODE1]] = !{
; CHECK-FIRST-SAME: i32 1
; CHECK-FIRST-SAME: i32 2

; CHECK-SECOND: ![[NODE2]] = !{i32 1}

; CHECK-THIRD: ![[NODE3]] = !{i32 2}
