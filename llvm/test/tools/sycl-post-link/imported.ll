; This test checks that -imports option correctly generates imported symbol files.
;
; RUN: sycl-post-link -symbols -imports -split=kernel -S < %s -o %t.table

; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM-0
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM-1
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-SYM-2

; RUN: FileCheck %s -input-file=%t_0.imported_sym --check-prefixes CHECK-IMPORTED-SYM-0
; RUN: FileCheck %s -input-file=%t_1.imported_sym --check-prefixes CHECK-IMPORTED-SYM-1
; RUN: FileCheck %s -input-file=%t_2.imported_sym --check-prefixes CHECK-IMPORTED-SYM-2

; CHECK-SYM-0: middle
; CHECK-IMPORTED-SYM-0: childD

; CHECK-SYM-1: foo
; CHECK-IMPORTED-SYM-1: childA
; CHECK-IMPORTED-SYM-1: childC
; CHECK-IMPORTED-SYM-1: childD

; CHECK-SYM-2: bar
; CHECK-IMPORTED-SYM-2: childB
; CHECK-IMPORTED-SYM-2: childC
; CHECK-IMPORTED-SYM-2: childD
; CHECK-IMPORTED-SYM-2: _Z7outsidev

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

  call void @_Z8__insidev()
  call void @_Z7outsidev()
  call void @_Z13_spirv_insidev()  
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
declare void @_Z13_spirv_insidev()

attributes #0 = { "sycl-module-id"="a.cpp" }
