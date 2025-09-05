; This test checks that the post-link tool doesn't incorrectly remove function
; declarations which are still in use while erasing the "llvm.used" global.
;
; RUN: sycl-post-link -properties -split=auto -symbols -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll
;
target triple = "spir64-unknown-unknown"

; CHECK-NOT: llvm.used
@llvm.used = appending global [2 x i8*] [i8* bitcast (void ()* @notused to i8*), i8* bitcast (void ()* @stillused to i8*)], section "llvm.metadata"

; CHECK: declare spir_func void @stillused
declare spir_func void @stillused() #0
declare spir_func void @notused() #0

define spir_kernel void @entry() #0 {
  call spir_func void @stillused()
  ret void
}

define spir_kernel void @bar() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="erase_used_decl.cpp" }
