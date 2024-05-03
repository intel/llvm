; This test locks down that assert functions are still noinline even if a
; genx_volatile global is present.
;
; RUN: opt < %s -passes=LowerESIMD -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::intel::esimd::simd" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" = type { <16 x float> }

@va = dso_local global %"class.sycl::_V1::ext::intel::esimd::simd" zeroinitializer, align 64 #0

define dso_local spir_func void @__assert_fail(ptr addrspace(4) %ptr) {
; CHECK: define dso_local spir_func void @__assert_fail(ptr addrspace(4) %ptr) #[[#ATTR:]] {
  ret void
}

define dso_local spir_func void @__devicelib_assert_fail(ptr addrspace(4) %ptr) {
; CHECK: define dso_local spir_func void @__devicelib_assert_fail(ptr addrspace(4) %ptr) #[[#ATTR]] {
  ret void
}

; CHECK: attributes #[[#ATTR]] = { noinline }
attributes #0 = { "genx_byte_offset"="192" "genx_volatile" }
!0 = !{}
