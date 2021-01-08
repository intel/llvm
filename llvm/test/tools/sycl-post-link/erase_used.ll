; This test checks that the post-link tool does not add "llvm.used" global to
; the output modules when splitting kernels.
;
; RUN: sycl-post-link -split=kernel -S %s -o %T/files.table
; RUN: FileCheck %s -input-file=%T/files_0.ll
; RUN: FileCheck %s -input-file=%T/files_1.ll

target triple = "spir64-unknown-unknown-sycldevice"

; CHECK-NOT: llvm.used
@llvm.used = appending global [2 x i8*] [i8* bitcast (void ()* @foo to i8*), i8* bitcast (void ()* @bar to i8*)], section "llvm.metadata"

define weak_odr spir_kernel void @foo() {
  ret void
}

define weak_odr spir_kernel void @bar() {
  ret void
}
