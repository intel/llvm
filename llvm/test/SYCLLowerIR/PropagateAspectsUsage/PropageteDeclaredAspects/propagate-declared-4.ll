; RUN: opt -passes=sycl-propagate-aspects-usage %s -S | FileCheck %s

target triple = "spir64-unknown-unknown"

; CHECK: void @kernel() !sycl_used_aspects ![[#ASPECT:]]
define weak_odr dso_local spir_kernel void @kernel() {
entry:
  call spir_func void @_Z3foov()
  ret void
}

; CHECK: !sycl_declared_aspects ![[#ASPECT]] !sycl_used_aspects ![[#ASPECT]] {{.*}} @_Z3foov()
declare !sycl_declared_aspects !1 dso_local spir_func void @_Z3foov()

!sycl_aspects = !{!0}

!0 = !{!"fp64", i32 6}
!1 = !{i32 2}
