; This test checks that LowerESIMD pass adds alwaysinline attribute to
; functions, except those marked with
; - spir_kernel
; - noinline
; - "VCStackCall"

; RUN: opt -passes=LowerESIMD -S < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function w/o attributes, must be marked with "alwaysinline"
define dso_local spir_func void @no_attrs_func(ptr addrspace(4) %ptr) {
; CHECK: define dso_local spir_func void @no_attrs_func(ptr addrspace(4) %ptr) #[[ATTRS1:[0-9]+]] {
  store float 2.0, ptr addrspace(4) %ptr
  ret void
}

; VCStackCall function, must not be marked with "alwaysinline"
define dso_local spir_func void @vc_stack_call_func(ptr addrspace(4) %ptr) #0 {
; CHECK: define dso_local spir_func void @vc_stack_call_func(ptr addrspace(4) %ptr) #[[ATTRS2:[0-9]+]] {
  store float 1.0, ptr addrspace(4) %ptr
  ret void
}

; Function with "noinline" attribute", must not be marked with "alwaysinline"
define dso_local spir_func void @noinline_func(ptr addrspace(4) %ptr) #1 {
; CHECK: define dso_local spir_func void @noinline_func(ptr addrspace(4) %ptr) #[[ATTRS3:[0-9]+]] {
  store float 2.0, ptr addrspace(4) %ptr
  ret void
}

; Kernel, must not be marked with "alwaysinline"
define dso_local spir_kernel void @KERNEL(ptr addrspace(4) %ptr) !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
; CHECK: define dso_local spir_kernel void @KERNEL(ptr addrspace(4) %ptr) #[[ATTRS4:[0-9]+]] !sycl_explicit_simd !{{.*}} !intel_reqd_sub_group_size !{{.*}} {
  store float 2.0, ptr addrspace(4) %ptr
  ret void
}

; Function with "noinline" attribute must be marked with "alwaysinline" if it is an ESIMD namespace function
define dso_local spir_func void @_ZNK4sycl3_V13ext5intel5esimd6detail13simd_obj_implIiLi16ENS3_4simdIiLi16EEEvE4dataEv(ptr addrspace(4) %ptr) #1 {
; CHECK: define dso_local spir_func void @_ZNK4sycl3_V13ext5intel5esimd6detail13simd_obj_implIiLi16ENS3_4simdIiLi16EEEvE4dataEv(ptr addrspace(4) %ptr) #[[ATTRS1]] {
ret void
}


attributes #0 = { "VCStackCall" }
attributes #1 = { noinline }
; CHECK-DAG: attributes #[[ATTRS1]] = { alwaysinline }
; CHECK-DAG: attributes #[[ATTRS2]] = { "VCStackCall" }
; CHECK-DAG: attributes #[[ATTRS3]] = { noinline }
; CHECK-DAG: attributes #[[ATTRS4]] = { "CMGenxMain" "oclrt"="1" }

!0 = !{}
!1 = !{i32 1}
