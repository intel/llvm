; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
;
; Test checks that the pass is able to propagate information about used aspects
; from simple composite types to functions and kernels which use them

; Optional
%A.optional = type { i32 }

; Not optional
%B.core = type { i32 }

; Not optional
%C.core = type { i32 }

%D1.contains.optional = type { %A.optional, %B.core, %C.core }

%D2.does.not.contain.optional = type { %B.core, %C.core }

%E.contains.optional = type { %B.core, %C.core, %D1.contains.optional }

%F1.points.to.optional = type { %B.core, %C.core*, %D1.contains.optional* }

%F2.does.not.contain.optional = type { %B.core, %C.core*, %D2.does.not.contain.optional* }

; CHECK: spir_kernel void @kernelD1.uses.optional() !sycl_used_aspects ![[MDID:[0-9]+]]
define spir_kernel void @kernelD1.uses.optional() {
  %tmp = alloca %D1.contains.optional
  ret void
}

; CHECK: spir_func void @funcD1.uses.optional() !sycl_used_aspects ![[MDID]] {
define spir_func void @funcD1.uses.optional() {
  %tmp = alloca %D1.contains.optional
  ret void
}

; CHECK: spir_kernel void @kernelD2.does.not.use.optional()
define spir_kernel void @kernelD2.does.not.use.optional() {
  %tmp = alloca %D2.does.not.contain.optional
  ret void
}

; CHECK: spir_func void @funcD2.does.not.use.optional() {
define spir_func void @funcD2.does.not.use.optional() {
  %tmp = alloca %D2.does.not.contain.optional
  ret void
}

; CHECK: spir_kernel void @kernelE.uses.optional() !sycl_used_aspects ![[MDID]]
define spir_kernel void @kernelE.uses.optional() {
  %tmp = alloca %E.contains.optional
  ret void
}

; CHECK: spir_func void @funcE.uses.optional() !sycl_used_aspects ![[MDID]] {
define spir_func void @funcE.uses.optional() {
  %tmp = alloca %E.contains.optional
  ret void
}

; CHECK: spir_kernel void @kernelF1.points.to.optional()
define spir_kernel void @kernelF1.points.to.optional() {
  %tmp = alloca %F1.points.to.optional
  ret void
}

; CHECK: spir_func void @funcF1.points.to.optional() {
define spir_func void @funcF1.points.to.optional() {
  %tmp = alloca %F1.points.to.optional
  ret void
}

; CHECK: spir_kernel void @kernelF2.does.not.use.optional()
define spir_kernel void @kernelF2.does.not.use.optional() {
  %tmp = alloca %F2.does.not.contain.optional
  ret void
}

; CHECK: spir_func void @funcF2.does.not.use.optional() {
define spir_func void @funcF2.does.not.use.optional() {
  %tmp = alloca %F2.does.not.contain.optional
  ret void
}

; CHECK: spir_func %A.optional @funcA.returns.optional() !sycl_used_aspects ![[MDID]] {
define spir_func %A.optional @funcA.returns.optional() {
  %tmp = alloca %A.optional
  %ret = load %A.optional, %A.optional* %tmp
  ret %A.optional %ret
}

; CHECK: spir_func void @funcA.uses.array.of.optional() !sycl_used_aspects ![[MDID]] {
define spir_func void @funcA.uses.array.of.optional() {
  %tmp = alloca [4 x %A.optional]
  ret void
}

; CHECK: spir_func void @funcA.assepts.optional(%A.optional %0) !sycl_used_aspects ![[MDID]] {
define spir_func void @funcA.assepts.optional(%A.optional %0) {
  ret void
}

!sycl_types_that_use_aspects = !{!0}
!0 = !{!"A.optional", i32 1}

!sycl_aspects = !{!1}
!1 = !{!"fp64", i32 6}

; CHECK: ![[MDID]] = !{i32 1}
