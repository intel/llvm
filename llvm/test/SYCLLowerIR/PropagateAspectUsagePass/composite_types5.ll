; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S | FileCheck %s
;
; Test checks different combinations of one aspect.

%A = type { i32 }
%B = type { i32 }
%C = type { %A, %B }

; CHECK-DAG: dso_local spir_kernel void @kernel1() !intel_used_aspects !2 {
define dso_local spir_kernel void @kernel1() {
  call spir_func void @func1()
  ret void
}

; CHECK-DAG: dso_local spir_func void @func1() !intel_used_aspects !2 {
define weak dso_local spir_func void @func1() {
  %tmp1 = alloca %A
  %tmp2 = alloca %B
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernel2() !intel_used_aspects !2 {
define weak dso_local spir_kernel void @kernel2() {
  %tmp = alloca %C
  ret void
}


; CHECK-DAG: dso_local spir_kernel void @kernel3() !intel_used_aspects !2 {
define weak dso_local spir_kernel void @kernel3() {
  call spir_func void @func2()
  call spir_func void @func3()
  ret void
}

; CHECK-DAG: dso_local spir_func void @func2() !intel_used_aspects !2 {
define weak dso_local spir_func void @func2() {
  %tmp = alloca %A
  ret void
}

; CHECK-DAG: dso_local spir_func void @func3() !intel_used_aspects !2 {
define weak dso_local spir_func void @func3() {
  %tmp = alloca %B
  ret void
}

!intel_types_that_use_aspects = !{!0, !1}
!0 = !{!"A", i32 1}
!1 = !{!"B", i32 1}

; CHECK-DAG: !2 = !{i32 1}
