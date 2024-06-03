; This test checks that the -emit-imported-symbols option generates a list of imported symbols
;
; RUN: sycl-post-link -symbols -emit-imported-symbols -split=kernel -S < %s -o %t.table

; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM-0
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM-1
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-SYM-2

;; Function names were chosen so that no function with a 'inside' in their function name is imported
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefixes CHECK-IMPORTED-SYM-0 --implicit-check-not='inside'
; RUN: FileCheck %s -input-file=%t_1.prop --check-prefixes CHECK-IMPORTED-SYM-1 --implicit-check-not='inside'
; RUN: FileCheck %s -input-file=%t_2.prop --check-prefixes CHECK-IMPORTED-SYM-2 --implicit-check-not='inside'

; CHECK-SYM-0: middle
; CHECK-IMPORTED-SYM-0: [SYCL/imported symbols]
; CHECK-IMPORTED-SYM-0-NEXT: childD

; CHECK-SYM-1: foo
; CHECK-IMPORTED-SYM-1: [SYCL/imported symbols]
; CHECK-IMPORTED-SYM-1-NEXT: childA
; CHECK-IMPORTED-SYM-1-NEXT: childC
; CHECK-IMPORTED-SYM-1-NEXT: childD

; CHECK-SYM-2: bar
; CHECK-IMPORTED-SYM-2: [SYCL/imported symbols]
; CHECK-IMPORTED-SYM-2-NEXT: childB
; CHECK-IMPORTED-SYM-2-NEXT: childC
; CHECK-IMPORTED-SYM-2-NEXT: childD
; CHECK-IMPORTED-SYM-2-NEXT: _Z7outsidev

target triple = "spir64-unknown-unknown"

; CHECK-NOT: llvm.used
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
