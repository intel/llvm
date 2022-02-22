; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S > %t.ll
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-FIRST
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-SECOND
;
; Test checks complex types graphs.

;     ----> A
;     |     |
;     |     v
;    *|     B
;     |    / \
;     |   v   v
;     E<--C   D
;     |   ^
;    *|   |*
;     v   |
;     F-->G
;       *
;

%A = type { %B }
%B = type { %C, %D }
%C = type { %E }
%D = type { i32 }
%E = type { %A*, %F* }
%F = type { %G* }
%G = type { %C* }


; CHECK-FIRST-DAG: dso_local spir_kernel void @kernel1() !intel_used_aspects ![[NODE1:[0-9]+]] {
define dso_local spir_kernel void @kernel1() {
  %tmp = alloca %A
  ret void
}

; CHECK-FIRST-DAG: dso_local spir_kernel void @kernel2() !intel_used_aspects ![[NODE1:[0-9]+]] {
define dso_local spir_kernel void @kernel2() {
  %tmp = alloca %C
  ret void
}

; CHECK-FIRST-DAG: dso_local spir_kernel void @kernel3() !intel_used_aspects ![[NODE1:[0-9]+]] {
define dso_local spir_kernel void @kernel3() {
  %tmp = alloca %F
  ret void
}

;     -----> A1
;     |      |
;     |      v
;    *|      B1
;     |     /
;     |    v
;     E1<--C1
;     |    ^
;    *|    |*
;     v    |
;     F1-->G1-->D1
;       *
;

%A1 = type { %B1 }
%B1 = type { %C1}
%C1 = type { %E1 }
%D1 = type { i32 }
%E1 = type { %A1*, %F1* }
%F1 = type { %G1* }
%G1 = type { %C1*, %D1 }


; CHECK-SECOND-DAG: dso_local spir_kernel void @kernel4() !intel_used_aspects ![[NODE2:[0-9]+]] {
define dso_local spir_kernel void @kernel4() {
  %tmp = alloca %A1
  ret void
}

!intel_types_that_use_aspects = !{!0, !1}
!0 = !{!"D", i32 1}
!1 = !{!"D1", i32 2}

; CHECK-FIRST-DAG: ![[NODE1]] = !{i32 1}
; CHECK-SECOND-DAG: ![[NODE2]] = !{i32 2}
