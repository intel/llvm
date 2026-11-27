; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
;
; Check the two-run model used by clang: fp16 is excluded from the first run
; and is only decided by the second run.
; RUN: opt -passes=sycl-propagate-aspects-usage -sycl-propagate-aspects-usage-exclude-aspects=fp16 < %s -S -o %t_first.ll
; RUN: FileCheck %s --input-file %t_first.ll --check-prefix=CHECK-FIRST
; RUN: opt -passes=sycl-propagate-aspects-usage < %t_first.ll -S | FileCheck %s
;
; Test checks that the fp16 aspect is attached to functions whose code has the
; LLVM half type, without any type annotated with the aspect. Code that only
; computes addresses of half values, or moves 16-bit values without a half
; type, does not use the aspect.

; CHECK-FIRST-NOT: !sycl_used_aspects

%composite = type { half }

; CHECK: spir_kernel void @kernel() {{.*}}!sycl_used_aspects ![[MDID:[0-9]+]]
define spir_kernel void @kernel() {
  call spir_func void @func()
  ret void
}

; CHECK: spir_func void @func() !sycl_used_aspects ![[MDID]] {
define spir_func void @func() {
  %tmp = alloca half
  ret void
}

; CHECK: spir_func void @func.array() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.array() {
  %tmp = alloca [4 x half]
  ret void
}

; CHECK: spir_func void @func.vector() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.vector() {
  %tmp = alloca <4 x half>
  ret void
}

; CHECK: spir_func void @func.composite() !sycl_used_aspects ![[MDID]] {
define spir_func void @func.composite() {
  %tmp = alloca %composite
  ret void
}

; CHECK: spir_func void @func.fadd(ptr %p) !sycl_used_aspects ![[MDID]] {
define spir_func void @func.fadd(ptr %p) {
  %a = load half, ptr %p
  %b = fadd half %a, %a
  store half %b, ptr %p
  ret void
}

; CHECK: spir_func i1 @func.fcmp(half %a, half %b) !sycl_used_aspects ![[MDID]] {
define spir_func i1 @func.fcmp(half %a, half %b) {
  %c = fcmp olt half %a, %b
  ret i1 %c
}

; CHECK: spir_func float @func.fpext(ptr %p) !sycl_used_aspects ![[MDID]] {
define spir_func float @func.fpext(ptr %p) {
  %a = load half, ptr %p
  %f = fpext half %a to float
  ret float %f
}

; CHECK: spir_func void @func.copy.memcpy(ptr %src, ptr %dst) {
define spir_func void @func.copy.memcpy(ptr %src, ptr %dst) {
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %dst, ptr align 2 %src, i64 2, i1 false)
  ret void
}

; CHECK: spir_func void @func.copy.i16(ptr %src, ptr %dst) {
define spir_func void @func.copy.i16(ptr %src, ptr %dst) {
  %v = load i16, ptr %src
  store i16 %v, ptr %dst
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

!sycl_aspects = !{!0, !1}
!0 = !{!"fp16", i32 5}
!1 = !{!"fp64", i32 6}

; CHECK: ![[MDID]] = !{i32 5}
