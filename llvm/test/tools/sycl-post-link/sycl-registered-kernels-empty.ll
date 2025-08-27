; RUN: sycl-post-link -properties %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop --implicit-check-not="[SYCL/registered kernels]"
!sycl_registered_kernels = !{}
