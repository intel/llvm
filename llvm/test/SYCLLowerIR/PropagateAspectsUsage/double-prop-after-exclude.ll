; RUN: opt -passes=sycl-propagate-aspects-usage -sycl-propagate-aspects-usage-exclude-aspects=fp64 < %s -S -o %t_first.ll
; RUN: opt -passes=sycl-propagate-aspects-usage < %t_first.ll -S -o %t_second.ll
; FileCheck %s --input-file %t_first.ll --check-prefix=CHECK-FIRST
; FileCheck %s --input-file %t_second.ll --check-prefix=CHECK-SECOND
;
; Test checks that fp64 usage is correctly propagate in the two-run model.

%composite = type { double }

; CHECK-FIRST-NOT: spir_kernel void @kernel() {{.*}} !sycl_used_aspects
; CHECK-SECOND: spir_kernel void @kernel() !sycl_used_aspects ![[MDID:]]
define spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK-FIRST-NOT: spir_func void @func() {{.*}} !sycl_used_aspects
; CHECK-SECOND: spir_func void @func() !sycl_used_aspects ![[MDID]] {
define spir_func void @func() {
  %tmp = alloca double
  ret void
}

; CHECK-FIRST-NOT: spir_func void @func.array() {{.*}} !sycl_used_aspects
; CHECK-SECOND: spir_func void @func.array() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.array() {
  %tmp = alloca [4 x double]
  ret void
}

; CHECK-FIRST-NOT: spir_func void @func.vector() {{.*}} !sycl_used_aspects
; CHECK-SECOND: spir_func void @func.vector() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.vector() {
  %tmp = alloca <4 x double>
  ret void
}

; CHECK-FIRST-NOT: spir_func void @func.composite() {{.*}} !sycl_used_aspects
; CHECK-SECOND: spir_func void @func.composite() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.composite() {
  %tmp = alloca %composite
  ret void
}

!sycl_aspects = !{!0}
!0 = !{!"fp64", i32 6}

; CHECK-SECOND: ![[MDID]] = !{i32 6}
