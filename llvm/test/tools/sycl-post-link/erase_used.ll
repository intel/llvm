; This test checks that the post-link tool does not add "llvm.used" global to
; the output modules when splitting modules, creating a single row table,
; and outputing IR only
;
; RUN: sycl-post-link -properties -split=kernel -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll
; RUN: FileCheck %s -input-file=%t.files_1.ll
;
; RUN: sycl-post-link -properties -S -split=auto -symbols -split-esimd -lower-esimd -O2 -spec-const=emulation < %s -o %t.out.table
; RUN: FileCheck %s --input-file=%t.out_0.ll
;
; RUN: sycl-post-link -S -split=auto -ir-output-only < %s -o %t.out_ir_only.ll
; RUN: FileCheck %s --input-file %t.out_ir_only.ll

target triple = "spir64-unknown-unknown"

; CHECK-NOT: llvm.used
@llvm.used = appending global [2 x i8*] [i8* bitcast (void ()* @foo to i8*), i8* bitcast (void ()* @bar to i8*)], section "llvm.metadata"

define weak_odr spir_kernel void @foo() #0 {
  ret void
}

define weak_odr spir_kernel void @bar() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
