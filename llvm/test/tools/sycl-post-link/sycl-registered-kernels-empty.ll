; Empty modules should not produce unused IR, property, or symbol files.
; RUN: rm -rf %t.dir
; RUN: mkdir %t.dir
; RUN: sycl-post-link -properties %s -o %t.dir/empty.table
; RUN: FileCheck %s -input-file=%t.dir/empty.table --check-prefix=EMPTY
; RUN: ls %t.dir | FileCheck %s --check-prefix=FILES
; RUN: sycl-post-link -properties -symbols -split=auto %s -o %t.dir/empty.table
; RUN: FileCheck %s -input-file=%t.dir/empty.table --check-prefix=EMPTY
; RUN: ls %t.dir | FileCheck %s --check-prefix=FILES
; FILES: empty.table
; FILES-NOT: {{.}}
; EMPTY: [Code|Properties{{(\|Symbols)?}}]
; EMPTY-NOT: {{.}}
!sycl_registered_kernels = !{}
