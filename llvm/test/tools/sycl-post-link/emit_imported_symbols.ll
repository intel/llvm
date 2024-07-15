; This test checks that the -emit-imported-symbols option generates a list of imported symbols
; Function names were chosen so that no function with a 'inside' in their function name is imported
;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Test with -split=kernel
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; RUN: sycl-post-link -properties -symbols -emit-imported-symbols -split=kernel -S < %s -o %t_kernel.table

; RUN: FileCheck %s -input-file=%t_kernel_0.sym --check-prefixes CHECK-KERNEL-SYM-0
; RUN: FileCheck %s -input-file=%t_kernel_1.sym --check-prefixes CHECK-KERNEL-SYM-1
; RUN: FileCheck %s -input-file=%t_kernel_2.sym --check-prefixes CHECK-KERNEL-SYM-2

; RUN: FileCheck %s -input-file=%t_kernel_0.prop --check-prefixes CHECK-KERNEL-IMPORTED-SYM-0
; RUN: FileCheck %s -input-file=%t_kernel_1.prop --check-prefixes CHECK-KERNEL-IMPORTED-SYM-1
; RUN: FileCheck %s -input-file=%t_kernel_2.prop --check-prefixes CHECK-KERNEL-IMPORTED-SYM-2

; CHECK-KERNEL-SYM-0: middle
; CHECK-KERNEL-IMPORTED-SYM-0: [SYCL/imported symbols]
; CHECK-KERNEL-IMPORTED-SYM-0-NEXT: childD
; CHECK-KERNEL-IMPORTED-SYM-0-EMPTY:

; CHECK-KERNEL-SYM-1: foo
; CHECK-KERNEL-IMPORTED-SYM-1: [SYCL/imported symbols]
; CHECK-KERNEL-IMPORTED-SYM-1-NEXT: childA
; CHECK-KERNEL-IMPORTED-SYM-1-NEXT: childC
; CHECK-KERNEL-IMPORTED-SYM-1-NEXT: childD
; CHECK-KERNEL-IMPORTED-SYM-1-EMPTY:


; CHECK-KERNEL-SYM-2: bar
; CHECK-KERNEL-IMPORTED-SYM-2: [SYCL/imported symbols]
; CHECK-KERNEL-IMPORTED-SYM-2-NEXT: childB
; CHECK-KERNEL-IMPORTED-SYM-2-NEXT: childC
; CHECK-KERNEL-IMPORTED-SYM-2-NEXT: childD
; CHECK-KERNEL-IMPORTED-SYM-2-NEXT: _Z7outsidev
; CHECK-KERNEL-IMPORTED-SYM-2-EMPTY:

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Test with -split=source
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; RUN: sycl-post-link -properties -symbols -emit-imported-symbols -split=source -S < %s -o %t_source.table
; RUN: FileCheck %s -input-file=%t_source_0.sym --check-prefixes CHECK-SOURCE-SYM-0
; RUN: FileCheck %s -input-file=%t_source_0.prop --check-prefixes CHECK-SOURCE-IMPORTED-SYM-0

; RUN: sycl-post-link -properties -symbols -emit-imported-symbols -split=source -S < %s -o %t_source.table -O0
; RUN: FileCheck %s -input-file=%t_source_0.sym --check-prefixes CHECK-SOURCE-SYM-0
; RUN: FileCheck %s -input-file=%t_source_0.prop --check-prefixes CHECK-SOURCE-IMPORTED-SYM-0

; CHECK-SOURCE-SYM-0-DAG: foo
; CHECK-SOURCE-SYM-0-DAG: bar
; CHECK-SOURCE-SYM-0-DAG: middle

; CHECK-SOURCE-IMPORTED-SYM-0: [SYCL/imported symbols]
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: childA
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: childB
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: childC
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: childD
; CHECK-SOURCE-IMPORTED-SYM-0-NEXT: _Z7outsidev
; CHECK-SOURCE-IMPORTED-SYM-0-EMPTY:

target triple = "spir64-unknown-unknown"

@llvm.used = appending global [2 x ptr] [ptr @foo, ptr @bar], section "llvm.metadata"

define weak_odr spir_kernel void @foo() #0 {
  call void @childA()
  call void @childC()
  call void @middle() 
  ret void
}

define weak_odr spir_kernel void @bar() #0 {
  ;; Functions that are not SYCL External (i.e. they have no sycl-module-id) cannot be imported
  call spir_func void @__itt_offload_wi_start_wrapper()

  call void @childB()
  call void @childC()
  call void @middle()
  ;; LLVM intrinsics cannot be imported
  %dummy = call i8 @llvm.bitreverse.i8(i8 0)
  ;; Functions with a demangled name prefixed with a '__' are not imported
  call void @_Z8__insidev()
  call void @_Z7outsidev()

  ;; Functions that are not SYCL External (i.e. they have no sycl-module-id) cannot be imported
  call spir_func void @__itt_offload_wi_finish_wrapper()
  ret void
}

define void @middle() #0 {
  call void @childD()
  ret void
}

declare void @childA() #1
declare void @childB() #1
declare void @childC() #1
declare void @childD() #1

declare void @_Z7outsidev() #1
;; Verify unused functions are not imported
declare void @insideUnusedFunction() #1
declare void @_Z8__insidev() #1
declare i8 @llvm.bitreverse.i8(i8)

declare spir_func void @__itt_offload_wi_start_wrapper()
declare spir_func void @__itt_offload_wi_finish_wrapper()

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="external.cpp" }
