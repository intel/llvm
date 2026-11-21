; RUN: opt -passes=record-sycl-aspect-names -S < %s | FileCheck %s
;
; Checks that !sycl_used_aspects and !sycl_declared_aspects operands that are
; already in the {!"name", i32 value} form (e.g. coming from linked-in device
; bitcode that has already been through this pass) are left unchanged, while
; plain integer operands in the same node are still converted.

; e.g. !sycl_used_aspects = !{!{!"A", i32 0}}
; =>   !sycl_used_aspects = !{!{!"A", i32 0}}
;
; e.g. !sycl_used_aspects = !{!{!"B", i32 1}, i32 0}
; =>   !sycl_used_aspects = !{!{!"B", i32 1}, !{!"A", i32 0}}

; CHECK: define spir_func void @funcA() !sycl_used_aspects ![[#fA:]] {
define spir_func void @funcA() !sycl_used_aspects !10 {
  ret void
}

; CHECK: define spir_func void @funcB() !sycl_used_aspects ![[#fB:]] {
define spir_func void @funcB() !sycl_used_aspects !11 {
  ret void
}

; CHECK: define spir_func void @funcC() !sycl_declared_aspects ![[#fC:]] {
define spir_func void @funcC() !sycl_declared_aspects !12 {
  ret void
}

; A pair whose value is not listed in !sycl_aspects is also kept as is.
; CHECK: define spir_func void @funcD() !sycl_used_aspects ![[#fD:]] {
define spir_func void @funcD() !sycl_used_aspects !13 {
  ret void
}

!sycl_aspects = !{!0, !1}
!0 = !{!"A", i32 0}
!1 = !{!"B", i32 1}
!2 = !{!"X", i32 9}

!10 = !{!0}
!11 = !{!1, i32 0}
!12 = !{!0, !1}
!13 = !{!2, i32 1}

; CHECK-DAG: ![[#mA:]] = !{!"A", i32 0}
; CHECK-DAG: ![[#mB:]] = !{!"B", i32 1}
; CHECK-DAG: ![[#mX:]] = !{!"X", i32 9}
; CHECK-DAG: ![[#fA]] = !{![[#mA]]}
; CHECK-DAG: ![[#fB]] = !{![[#mB]], ![[#mA]]}
; CHECK-DAG: ![[#fC]] = !{![[#mA]], ![[#mB]]}
; CHECK-DAG: ![[#fD]] = !{![[#mX]], ![[#mB]]}
