; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S | FileCheck %s
;
; Test checks structures with pointers.

; In this test, the only type which is explicitly marked as optional is %D
; Various test cases check if the pass is able to deal with various
; type hierarchies and correctly propagate aspects.
;
; %A and %B are optional because they contain %D.
; %C contains a pointer on %A. Because of that there is a path
; to %D which means that %C is optional as well.
;
;  -->  A
;  |    |
; *|    B
;  |   / \
;  ---C   D

%A = type { %B }
%B = type { %C, %D}
%C = type { %A* }
%D = type { i32 }

; CHECK-DAG: dso_local spir_kernel void @kernel1() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel1() {
  %tmp = alloca %A
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernel2() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel2() {
  %tmp = alloca %C
  ret void
}

;  -->  A1
;  |    |
; *|    B1
; *|   / \
;  ---C1  D

%A1 = type { %B1 }
%B1 = type { %C1, %D}
%C1 = type { %A1** }

; CHECK-DAG: dso_local spir_kernel void @kernel3() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel3() {
  %tmp = alloca %A1
  ret void
}

; CHECK-DAG: dso_local spir_kernel void @kernel4() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel4() {
  %tmp = alloca %C1
  ret void
}


;     *
;   Z--->Y
;   ^   /
;  *|  /*
;   | /
;   X<

%Z = type { %Y* }
%Y = type { %X* }
%X = type { %Z* }

; CHECK-DAG: dso_local spir_kernel void @kernel5() {
define dso_local spir_kernel void @kernel5() {
  %tmp = alloca %Z
  ret void
}

;      *
;   Z1---->Y1
;   ^    / |
;  *|   /* |
;   |  /   v
;   X1<    D

%Z1 = type { %Y1* }
%Y1 = type { %X1*, %D }
%X1 = type { %Z1* }

; CHECK-DAG: dso_local spir_kernel void @kernel6() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel6() {
  %tmp = alloca %X1
  ret void
}

; simple loop
%L = type { %L* }

; CHECK-DAG: dso_local spir_kernel void @kernel7() {
define dso_local spir_kernel void @kernel7() {
  %tmp = alloca %L
  ret void
}

; Check a link by vector type
;  -->  A2
;  |    |
;  |    B2
; *|   / \
;  ---C2  D

%A2 = type { %B2 }
%B2 = type { %C2, %D}
%C2 = type { <4 x %A2*> }

; CHECK-DAG: dso_local spir_kernel void @kernel8() !intel_used_aspects !1 {
define dso_local spir_kernel void @kernel8() {
  %tmp2 = alloca %C2
  ret void
}

!intel_types_that_use_aspects = !{!0}
!0 = !{!"D", i32 1}

; CHECK-DAG: !1 = !{i32 1}
