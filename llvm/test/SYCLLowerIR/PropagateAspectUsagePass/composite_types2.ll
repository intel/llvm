; RUN: opt -passes=sycl-propagate-aspect-usage < %s -S > %t.ll
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-A
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-B
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-C
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-DE
;
; Test checks simple composite structures.

%A = type { %B, %C }

%B = type { i32 }

%C = type { i32 }

; CHECK-A: dso_local spir_kernel void @kernelA() !intel_used_aspects ![[NODE_A:[0-9]+]] {
define dso_local spir_kernel void @kernelA() {
  %tmp = alloca %A
  ret void
}

; CHECK-B: dso_local spir_kernel void @kernelB() !intel_used_aspects ![[NODE_B:[0-9]+]] {
define dso_local spir_kernel void @kernelB() {
  %tmp = alloca %B
  ret void
}

; CHECK-C: dso_local spir_kernel void @kernelC() !intel_used_aspects ![[NODE_C:[0-9]+]] {
define dso_local spir_kernel void @kernelC() {
  %tmp = alloca %C
  ret void
}

; CHECK-DE: dso_local spir_kernel void @kernelD() !intel_used_aspects ![[NODE_DE:[0-9]+]] {
define dso_local spir_kernel void @kernelD() {
  %tmp = alloca <4 x double>
  ret void
}

; CHECK-DE: dso_local spir_kernel void @kernelE() !intel_used_aspects ![[NODE_DE:[0-9]+]] {
define dso_local spir_kernel void @kernelE() {
  %tmp = alloca [4 x double]
  ret void
}

!intel_types_that_use_aspects = !{!0, !1}
!0 = !{!"B", i32 1}
!1 = !{!"C", i32 2}

; Check metadata which should appear
; CHECK-B: ![[NODE_B]] = !{i32 1}

; CHECK-C: ![[NODE_C]] = !{i32 2}

; CHECK-A: ![[NODE_A]] = !{{[{]}}{{i32 1, i32 2|i32 2, i32 1}}{{[}]}}

; CHECK-DE: ![[NODE_DE]] = !{i32 6}
