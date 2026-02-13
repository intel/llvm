; RUN: sycl-post-link -split=source -S < %s -o %t.table
; RUN: FileCheck %s --input-file=%t_0.ll --check-prefixes CHECK-SOURCE0
; RUN: FileCheck %s --input-file=%t_1.ll --check-prefixes CHECK-SOURCE1
;
; RUN: sycl-post-link -split=kernel -S < %s -o %t2.table
; RUN: FileCheck %s --input-file=%t2_0.ll --check-prefix CHECK-KERNEL0
; RUN: FileCheck %s --input-file=%t2_1.ll --check-prefix CHECK-KERNEL1
; RUN: FileCheck %s --input-file=%t2_2.ll --check-prefix CHECK-KERNEL2
;
; sycl-post-link performs internalization step after split phase. That split
; removes all functions which are not considered to be entry points.

target triple = "spir64-unknown-unknown"

; This function has no sycl-module-id attribute (i.e. it was not marked as
; SYCL_EXTERNAL) and therefore it is not considered to be an entry point and
; should always be internalized.
; CHECK-SOURCE0: define internal spir_func void @internal_func
; CHECK-SOURCE1: define internal spir_func void @internal_func
; CHECK-KERNEL1: define internal spir_func void @internal_func
; CHECK-KERNEL2: define internal spir_func void @internal_func
define spir_func void @internal_func() {
  ret void
}

; This function has sycl-module-id attribute (i.e. it was marked as
; SYCL_EXTERNAL) and therefore it is considered to be an entry point and should
; not be internalized.
; CHECK-SOURCE1: define spir_func void @sycl_external_func
; CHECK-KERNEL0: define spir_func void @sycl_external_func

; However, when this function is called from a kernel that is outlined into a
; separate device image, it is copied into that device image as a depenency and
; it is not considered to be an entry point there, i.e. it should be
; intenralized.
; CHECK-SOURCE0: define internal spir_func void @sycl_external_func
; CHECK-KERNEL1: define internal spir_func void @sycl_external_func
; CHECK-KERNEL2: define internal spir_func void @sycl_external_func
define spir_func void @sycl_external_func() #0 {
  ret void
}

; Kernels are always considered to be entry points
; CHECK-SOURCE1: define spir_kernel void @kernelA
; CHECK-KERNEL2: define spir_kernel void @kernelA
define spir_kernel void @kernelA() #0 {
  call void @internal_func()
  call void @sycl_external_func()
  ret void
}

; CHECK-SOURCE0: define spir_kernel void @kernelB
; CHECK-KERNEL1: define spir_kernel void @kernelB
define spir_kernel void @kernelB() #1 {
  call void @internal_func()
  call void @sycl_external_func()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="b.cpp" }
