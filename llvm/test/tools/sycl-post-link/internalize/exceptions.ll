; RUN: sycl-post-link -split=source -emit-only-kernels-as-entry-points -S < %s -o %t.table
; RUN: FileCheck %s --input-file=%t_0.ll --check-prefixes CHECK-SOURCE0
; RUN: FileCheck %s --input-file=%t_1.ll --check-prefixes CHECK-SOURCE1
; RUN: FileCheck %s --input-file=%t_2.ll --check-prefixes CHECK-SOURCE2
;
; RUN: sycl-post-link -split=kernel -emit-only-kernels-as-entry-points -S < %s -o %t2.table
; RUN: FileCheck %s --input-file=%t2_0.ll --check-prefix CHECK-KERNEL0
; RUN: FileCheck %s --input-file=%t2_1.ll --check-prefix CHECK-KERNEL1
; RUN: FileCheck %s --input-file=%t2_2.ll --check-prefix CHECK-KERNEL2
;
; sycl-post-link performs internalization step after split phase. That split
; removes all functions which are not considered to be entry points.

target triple = "spir64-unknown-unknown"

; This function has no sycl-module-id attribute (i.e. it was not marked as
; SYCL_EXTERNAL) and therefore it is not considered to be an entry point.
; However, it is marked with referenced-indirectly attribute which makes it an
; exception that should never be internalized.
; CHECK-SOURCE0: define spir_func void @referenced_indirectly_func
; CHECK-SOURCE2: define spir_func void @referenced_indirectly_func
; CHECK-KERNEL0: define spir_func void @referenced_indirectly_func
; CHECK-KERNEL1: define spir_func void @referenced_indirectly_func
define spir_func void @referenced_indirectly_func() #2 {
  ret void
}

; Even though this function has sycl-module-id attribute (i.e. it was marked as
; SYCL_EXTERNAL) it is not considered to be an entry point due to a flag passed
; to sycl-post-link.
; However, it is marked with indirectly-callable attribute which makes it an
; exception that should never be internalized. It is also outlined into a
; separate device image based on that attribute, all other device images should
; only have its declaration.
; CHECK-SOURCE0: declare spir_func void @indirectly_callable_func
; CHECK-SOURCE1: define spir_func void @indirectly_callable_func
; CHECK-SOURCE2: declare spir_func void @indirectly_callable_func
; CHECK-KERNEL0: declare spir_func void @indirectly_callable_func
; CHECK-KERNEL1: declare spir_func void @indirectly_callable_func
; CHECK-KERNEL2: define spir_func void @indirectly_callable_func
define spir_func void @indirectly_callable_func() #3 {
  ret void
}

; Kernels are always considered to be entry points
; CHECK-SOURCE2: define spir_kernel void @kernelA
; CHECK-KERNEL1: define spir_kernel void @kernelA
define spir_kernel void @kernelA() #0 {
  call void @referenced_indirectly_func()
  call void @indirectly_callable_func()
  ret void
}

; CHECK-SOURCE0: define spir_kernel void @kernelB
; CHECK-KERNEL0: define spir_kernel void @kernelB
define spir_kernel void @kernelB() #1 {
  call void @referenced_indirectly_func()
  call void @indirectly_callable_func()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="b.cpp" }
attributes #2 = { "referenced-indirectly" }
attributes #3 = { "sycl-module-id"="a.cpp" "indirectly-callable"="set-a" }

