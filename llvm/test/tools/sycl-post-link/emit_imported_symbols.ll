; This test checks that the -emit-imported-symbols option generates a list of imported symbols
; Function names were chosen so that no function with a 'inside' in their function name is imported
;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Test with -split=kernel
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; RUN: rm -f %t*.prop %t*.sym

; RUN: sycl-post-link -symbols -emit-imported-symbols -split=kernel -S < %s -o %t_kernel.table

; RUN: FileCheck %s -input-file=%t_kernel_0.sym --check-prefixes CHECK-KERNEL-SYM-0
; RUN: FileCheck %s -input-file=%t_kernel_1.sym --check-prefixes CHECK-KERNEL-SYM-1
; RUN: FileCheck %s -input-file=%t_kernel_2.sym --check-prefixes CHECK-KERNEL-SYM-2

; RUN: FileCheck %s -input-file=%t_kernel_0.prop --check-prefixes CHECK-KERNEL-IMPORTED-SYM-0 --implicit-check-not='inside'
; RUN: FileCheck %s -input-file=%t_kernel_1.prop --check-prefixes CHECK-KERNEL-IMPORTED-SYM-1 --implicit-check-not='inside'
; RUN: FileCheck %s -input-file=%t_kernel_2.prop --check-prefixes CHECK-KERNEL-IMPORTED-SYM-2 --implicit-check-not='inside'

; CHECK-KERNEL-SYM-0: middle
; CHECK-KERNEL-IMPORTED-SYM-0: [SYCL/imported symbols]
; CHECK-KERNEL-IMPORTED-SYM-0-NEXT: childD

; CHECK-KERNEL-SYM-1: foo
; CHECK-KERNEL-IMPORTED-SYM-1: [SYCL/imported symbols]
; CHECK-KERNEL-IMPORTED-SYM-1-NEXT: childA
; CHECK-KERNEL-IMPORTED-SYM-1-NEXT: childC
; CHECK-KERNEL-IMPORTED-SYM-1-NEXT: childD

; CHECK-KERNEL-SYM-2: bar
; CHECK-KERNEL-IMPORTED-SYM-2: [SYCL/imported symbols]
; CHECK-KERNEL-IMPORTED-SYM-2-NEXT: childB
; CHECK-KERNEL-IMPORTED-SYM-2-NEXT: childC
; CHECK-KERNEL-IMPORTED-SYM-2-NEXT: childD
; CHECK-KERNEL-IMPORTED-SYM-2-NEXT: _Z7outsidev

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Test with -split=source
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; RUN: sycl-post-link -symbols -emit-imported-symbols -split=source -S < %s -o %t_source.table

; RUN: FileCheck %s -input-file=%t_source_0.sym --check-prefixes CHECK-SOURCE-SYM-0

; RUN: FileCheck %s -input-file=%t_source_0.prop --check-prefixes CHECK-SOURCE-IMPORTED-SYM-0 --implicit-check-not='inside'

; CHECK-SOURCE-SYM-0-DAG: foo
; CHECK-SOURCE-SYM-0-DAG: bar
; CHECK-SOURCE-SYM-0-DAG: middle

; CHECK-SOURCE-IMPORTED-SYM-0: [SYCL/imported symbols]
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: childA
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: childB
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: childC
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: childD
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: _Z7outsidev

target triple = "spir64-unknown-unknown"

@llvm.used = appending global [2 x ptr] [ptr @foo, ptr @bar], section "llvm.metadata"

define weak_odr spir_kernel void @foo() #0 {
  call void @childA()
  call void @childC()
  call void @middle() 
  ret void
}

define weak_odr spir_kernel void @bar() #0 {
  call void @childB()
  call void @childC()
  call void @middle()

  ;; Functions with a demangled name prefixed with a '__' are not imported
  call void @_Z8__insidev()
  call void @_Z7outsidev()
  ret void
}

define void @middle() #0 {
  call void @childD() 
  ret void
}

declare void @childA()
declare void @childB()
declare void @childC()
declare void @childD()

declare void @_Z8__insidev()
declare void @_Z7outsidev()

attributes #0 = { "sycl-module-id"="a.cpp" }
