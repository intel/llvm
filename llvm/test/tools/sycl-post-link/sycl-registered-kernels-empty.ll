; RUN: sycl-post-link -properties %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop --implicit-check-not="[SYCL/registered kernels]"
; RUN: FileCheck %s -input-file=%t.table --check-prefix=EMPTY
; RUN: sycl-post-link -properties -symbols -split=auto %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table --check-prefix=EMPTY
; EMPTY: [Code|Properties{{(\|Symbols)?}}]
; EMPTY-NOT: {{.}}
!sycl_registered_kernels = !{}
